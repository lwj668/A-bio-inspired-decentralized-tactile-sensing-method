from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Optional, List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

# ---------------- 脉冲数据结构 ----------------
@dataclass
class Pulse:
    start: int
    end: int
    length: int
    amp_sign: int    # +1 / -1
    peak_val: float

@dataclass
class DecodeInfo:
    node_id: int
    dimI: int
    bits: np.ndarray
    startLeadP: Pulse
    endLeadP: Pulse
    bitPulses: List[Pulse]

# ---------------- 主函数 ----------------
def main():
    fs        = 200e6  # MHz
    amp       = 0.4
    num_bits  = 10
    num_nodes = 500  # 参数
    n_dim     = 3000

    T_lead_us  = 0.05 # 单位微秒， 参数
    lead_samps = int(round(T_lead_us * 1e-6 * fs))
    T_comp_us  = 0.5 # 单位微秒，参数
    comp_samps = int(round(T_comp_us * 1e-6 * fs))

    comp_tol = max(0, int(round(0.001 * comp_samps)))   # 参数
    lead_tol = max(0, int(round(0.0001 * lead_samps)))  # 参数

    spacing_ideal = n_dim * comp_samps
    spacing_tol   = int(round(0.005 * comp_samps))      # 参数

    max_offset_us    = 5000
    max_offset_samps = int(round(max_offset_us * 1e-6 * fs))

    rng = np.random.default_rng()
    if num_nodes > n_dim:
        raise ValueError("节点数量不能大于维度")

    # ------- 用 n_dim 拆成阵列 & 生成图案 -------
    rows, cols = best_grid(n_dim)
    # 大致勾一只“飞鸟”轮廓（示例，按你热图的方位再微调）
    # 1) 读图并转成 0..1 的灰度 img_mask01
    img_path = "浙大校标.png"  # 你的素材路径
    img = Image.open(img_path).convert("L")  # 灰度
    img = np.asarray(img, dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)  # 归一化到 0..1

    # 2) 直接调用 make_pattern(kind="image")
    # rows, cols = 64, 64  # 你的阵列尺寸
    pattern_logo = make_pattern(
        rows=rows, cols=cols,
        kind="image",
        center=(0.5, 0.5),  # 放在中心
        radius=0.38,  # 控制贴图大小（可调 0.25~0.45）
        angle_deg=0,  # 旋转角（可调）
        img_mask01=img,  # 来自上面的灰度图
        img_threshold=0.70,  # 白底(≈1)与蓝色(≈0.3~0.6)分离阈值
        invert=True  # 反相：让蓝色区域成为“内部=1”
    )

    # (可选) 让边缘更柔和、像真实按压分布
    pattern_soft = pattern_logo * make_pattern(
        rows=rows, cols=cols,
        kind="gaussian_finger",
        center=(0.5, 0.5),
        sigma=0.22,  # 越大越“柔”
        aspect=1.1
    )
    pattern_soft = (pattern_soft - pattern_soft.min()) / (pattern_soft.max() - pattern_soft.min() + 1e-12)

    # pattern = make_pattern(
    #     rows, cols,
    #     kind="gaussian_finger",            # "circle" | "ring" | "gaussian_finger" | "ellipse" |"square" |"zju_star"
    #     center=(0.5, 0.5),
    #     radius=0.45,            # 外径（circle/ring/ellipse）
    #     r_inner=0.145,           # 内径（ring）
    #     sigma=0.1,             # 手指高斯的扩散
    #     angle_deg=0,           # 旋转角度（手指/椭圆）
    #     aspect=0.8              # 纵横比（手指/椭圆）
    # )

    # ------- 从图案中指派 num_nodes 个节点（dimI 与 bits） -------
    dims_sel, bits_list, (r_idx, c_idx), vals_q = assign_nodes_from_pattern(
        pattern_soft, num_nodes, num_bits, pick_mode="topk"   # 或 "weighted"
    )

    all_dims = np.arange(1, n_dim + 1, dtype=int)
    # 先把选中的 dims 放在最前面，剩余的随机排列在后
    mask_sel = np.zeros_like(all_dims, dtype=bool)
    mask_sel[dims_sel - 1] = True
    rest_dims = all_dims[~mask_sel]
    rng.shuffle(rest_dims)
    dim_array = np.concatenate([dims_sel, rest_dims])  # 前 num_nodes 个即我们需要的

    bits_cell: List[np.ndarray] = [None] * num_nodes
    for i in range(num_nodes):
        bits_cell[i] = bits_list[i]  # 直接用目标强度转换的二进制

    # -------构造各节点波形并叠加 -------
    frame_cell: List[np.ndarray] = []
    frame_len = np.zeros(num_nodes, dtype=np.int64)
    for i in range(num_nodes):
        dimI = int(dim_array[i])  # 我们保证前 num_nodes 个维度就是图案选出的
        w = build_one_node_frame_updated(bits_cell[i], dimI, n_dim, comp_samps, lead_samps, amp)
        frame_cell.append(w)
        frame_len[i] = w.size

    big_len = int(frame_len.max()) + max_offset_samps + 1000
    wave_sub = np.zeros(big_len, dtype=np.float32)
    subset = np.arange(num_nodes, dtype=int)  # 此处就是全部参与
    offsets = rng.integers(0, max_offset_samps + 1, size=num_nodes, dtype=np.int64)

    for idx in subset:
        w = frame_cell[idx]
        off = int(offsets[idx])
        wave_sub[off:off+w.size] += w

    # target_snr_db = 20
    # wave_sub, snr_db_meas, noise_sigma = add_awgn(wave_sub, snr_db=target_snr_db, rng=rng, dtype=np.float32)

    # 解码计时
    T0 = time.perf_counter()
    decoded, pulses, leads, bitsPulses, bitGroups, stats, _rise_left, _fall_left = decode_all_nodes_fast_v2(
        wave_sub, fs, amp, num_bits, n_dim, comp_samps, lead_samps, comp_tol, lead_tol, spacing_ideal, spacing_tol
    )
    T1 = time.perf_counter()

    print(f"[Stats] events={stats['t_events']:.3f}s, group={stats['t_group']:.3f}s, leads={stats['t_leads']:.3f}s, total={T1-T0:.3f}s")
    print(f"[Counts] pulses={stats['n_pulses']}, leads={stats['n_leads']}, bits={stats['n_bits']}, groups={stats['n_groups']}, decoded={stats['n_decoded']}")

    # 差分重构（O(N + #pulses)）
    diff = np.zeros(wave_sub.size + 1, dtype=np.float32)
    for rec in decoded:
        for p in [rec.startLeadP] + rec.bitPulses + [rec.endLeadP]:
            val = p.amp_sign * amp
            diff[p.start] += val
            diff[p.end + 1] -= val
    wave_recon = np.cumsum(diff[:-1])
    wave_resid = wave_sub - wave_recon

    # BER 评估
    subset_idx = subset
    total_bits = subset_idx.size * num_bits
    error_bits = 0
    used_dec = np.zeros(len(decoded), dtype=bool)

    print("===== 误码率评估：=====")
    for node_idx in subset_idx:
        realDim = int(dim_array[node_idx])
        realBits = bits_cell[node_idx]
        k = -1
        for j, rec in enumerate(decoded):
            if (not used_dec[j]) and rec.dimI == realDim:
                k = j; break
        if k < 0:
            error_bits += num_bits
        else:
            used_dec[k] = True
            decBits = decoded[k].bits
            nErr = int((realBits != decBits).sum())
            error_bits += nErr

    BER = error_bits / total_bits if total_bits > 0 else 0.0
    print(f"*** 统计 => error_bits={error_bits}, total_bits={total_bits}, BER={BER:.4g}")

    # ========= 传感器阵列 热图可视化 =========
    sim_heat, dec_heat, diff_heat, (R, C) = build_sensor_heatmaps(
        bits_cell=bits_cell,
        dim_array=dim_array,
        subset_idx=subset,
        decoded_info=decoded,
        num_bits=num_bits,
        n_dim=n_dim
    )
    plot_sensor_heatmaps(sim_heat, dec_heat, diff_heat, num_bits,
                         title_prefix=f"Pattern: ring | array {R}x{C} | nodes={num_nodes}")

    # ----------------可视化剩余边沿事件 ----------------
    # edge_only_wave = build_unmatched_edge_signal(_rise_left, _fall_left, amp, wave_sub.size)
    # plot_unmatched_edge_signal(edge_only_wave, _rise_left, _fall_left,
    #                            title="Unmatched-edge-only signal")

    # time_all = np.arange(big_len) / fs * 1e3  # ms
    # plt.figure(figsize=(12, 4))
    # plt.plot(time_all, wave_sub, linewidth=1)
    # plt.xlabel('time (ms)');
    # plt.ylabel('amplitude (V)')
    # plt.title('signal')
    # plt.grid(True)
    # plt.show()
    #
    # plt.figure(figsize=(12, 4))
    # plt.plot(time_all, wave_recon, linewidth=1)
    # plt.xlabel('time (ms)');
    # plt.ylabel('amplitude (V)')
    # plt.title('reconstruct signal')
    # plt.grid(True)
    # plt.show()
    #
    # plt.figure(figsize=(12, 4))
    # plt.plot(time_all, wave_resid, linewidth=1)
    # plt.xlabel('time (ms)'); plt.ylabel('amplitude (V)')
    # plt.title('res signal')
    # plt.grid(True)
    # plt.show()

# ---------------- 编码信号生成 ----------------
def build_one_node_frame_updated(bits: np.ndarray,
                                 dimI: int,
                                 n_dim: int,
                                 comp_samps: int,
                                 lead_samps: int,
                                 amp: float) -> np.ndarray:
    num_bits = bits.size
    if num_bits != 10:
        raise ValueError("示例假定每节点10比特。")
    bits_area_9_len = 9 * n_dim * comp_samps
    bit10_len       = dimI * comp_samps
    gap_len         = (dimI - 1) * comp_samps
    frame_len = lead_samps + bits_area_9_len + bit10_len + gap_len + lead_samps
    frame = np.zeros(frame_len, dtype=np.float32)

    pos = 0
    frame[pos:pos+lead_samps] = +amp
    pos += lead_samps

    offset_in_bit = (dimI - 1) * comp_samps
    for b in range(9):
        st = pos + offset_in_bit
        ed = st + comp_samps
        frame[st:ed] = (+amp if bits[b] == 1 else -amp)
        pos += n_dim * comp_samps

    st10 = pos + (dimI - 1) * comp_samps
    ed10 = st10 + comp_samps
    frame[st10:ed10] = (+amp if bits[9] == 1 else -amp)
    pos += bit10_len
    pos += gap_len

    frame[pos:pos+lead_samps] = +amp
    pos += lead_samps
    assert pos == frame_len
    return frame

# ---------------- 总解码函数 ----------------
def decode_all_nodes_fast_v2(wave: np.ndarray,
                             fs: float,
                             amp: float,
                             num_bits: int,
                             n_dim: int,
                             comp_samps: int,
                             lead_samps: int,
                             comp_tol: int,
                             lead_tol: int,
                             spacing_ideal: int,
                             spacing_tol: int,
                             ):
    wave = wave.astype(np.float32).ravel()

    # 脉冲检测
    t0 = time.perf_counter()
    pulses, _rise_left, _fall_left = detect_pulses_multiset(wave, amp, comp_samps, lead_samps, comp_tol, lead_tol)
    t1 = time.perf_counter()

    # 分类先导/比特
    lead_tol = int(round(0.1 * lead_samps))
    leads: List[Pulse] = []
    bits:  List[Pulse] = []
    for p in pulses:
        if (p.amp_sign > 0) and (abs(p.length - lead_samps) <= lead_tol):
            leads.append(p)
        else:
            bits.append(p)

    # 10脉冲分组
    bits.sort(key=lambda p: p.start)
    bit_groups, used_bits = group_bits_fast(bits, num_bits, spacing_ideal, spacing_tol)
    t2 = time.perf_counter()

    # 先导匹配
    leads.sort(key=lambda p: p.start)
    lead_used = np.zeros(len(leads), dtype=bool)
    group_candidates: List[List[Tuple[int,int,int,int]]] = []

    for idxs in bit_groups:
        firstP = bits[idxs[0]]
        lastP  = bits[idxs[-1]]
        pairs = match_leads_min_diff(leads, lead_used, firstP.start, lastP.end)
        group_candidates.append(pairs)

    matched_groups = np.zeros(len(bit_groups), dtype=bool)
    decoded: List[DecodeInfo] = []

    # 阶段一：唯一候选直接落定
    for g, pairs in enumerate(group_candidates):
        if matched_groups[g]: continue
        if len(pairs) == 1:
            s_idx, e_idx, dist1, _dist2 = pairs[0]
            if (not lead_used[s_idx]) and (not lead_used[e_idx]):
                lead_used[s_idx] = True
                lead_used[e_idx] = True
                dimI_approx = int(np.clip(int(round(dist1 / comp_samps)) + 1, 1, n_dim))
                groupP = [bits[k] for k in bit_groups[g]]
                bits_dec = np.array([1 if p.amp_sign > 0 else 0 for p in groupP], dtype=np.int8)
                info = DecodeInfo(node_id=g+1, dimI=dimI_approx, bits=bits_dec,
                                  startLeadP=leads[s_idx], endLeadP=leads[e_idx], bitPulses=groupP)
                decoded.append(info)
                matched_groups[g] = True

    # 阶段二：迭代剔除冲突，直到出现唯一候选
    changed = True
    while changed:
        changed = False
        for g, pairs in enumerate(group_candidates):
            if matched_groups[g] or not pairs: continue
            valid = [(s,e,d1,d2) for (s,e,d1,d2) in pairs if (not lead_used[s]) and (not lead_used[e])]
            group_candidates[g] = valid
            if len(valid) == 1:
                s_idx, e_idx, dist1, _dist2 = valid[0]
                if (not lead_used[s_idx]) and (not lead_used[e_idx]):
                    lead_used[s_idx] = True
                    lead_used[e_idx] = True
                    dimI_approx = int(np.clip(int(round(dist1 / comp_samps)) + 1, 1, n_dim))
                    groupP = [bits[k] for k in bit_groups[g]]
                    bits_dec = np.array([1 if p.amp_sign > 0 else 0 for p in groupP], dtype=np.int8)
                    info = DecodeInfo(node_id=g+1, dimI=dimI_approx, bits=bits_dec,
                                      startLeadP=leads[s_idx], endLeadP=leads[e_idx], bitPulses=groupP)
                    decoded.append(info)
                    matched_groups[g] = True
                    changed = True

    t3 = time.perf_counter()
    stats = dict(
        t_events=t1-t0,
        t_group=t2-t1,
        t_leads=t3-t2,
        n_pulses=len(pulses),
        n_leads=len(leads),
        n_bits=len(bits),
        n_groups=len(bit_groups),
        n_decoded=len(decoded),
    )
    return decoded, pulses, leads, bits, bit_groups, stats, _rise_left, _fall_left

# ---------------- 事件->脉冲----------------
def detect_pulses_multiset(signal: np.ndarray,
                           amp_base: float,
                           comp_samps: int,
                           lead_samps: int,
                           comp_tol: int,
                           lead_tol: int,
                           ) -> Tuple[List[Pulse], Dict[int,int], Dict[int,int]]:

    # ---------------- 参数与容差 ----------------
    total_tol = max(comp_tol, lead_tol)

    # 正向长度集合
    POS_SINGLE = [(lead_samps, 'L', lead_tol),
                  (comp_samps, 'C', comp_tol)]
    POS_COMPOSITE = [(comp_samps + lead_samps, 'CL', total_tol),
                     (2 * comp_samps, 'CC', comp_tol),
                     (2 * lead_samps, 'LL', lead_tol)]
    # 反向长度集合
    NEG_SINGLE = [(comp_samps, 'NC', comp_tol)]           # 负比特
    NEG_COMPOSITE = [(comp_samps - lead_samps, 'NCL', total_tol),
                     (2 * comp_samps, 'NCC', comp_tol)]

    # ---------------- 量化 -> 事件 ----------------
    c = np.rint(signal / amp_base).astype(np.int16)
    delta = np.diff(c)
    idx = np.nonzero(delta != 0)[0]

    rise_nodes: List[int] = []   # 上升沿列表
    fall_nodes: List[int] = []   # 下降沿列表
    for i in idx:
        pos = i + 1
        dv = int(delta[i])
        if dv > 0:
            rise_nodes.extend([pos] * dv)
        else:
            fall_nodes.extend([pos] * (-dv))

    nL = len(rise_nodes)   # 左侧节点数（上升沿）
    nR = len(fall_nodes)   # 右侧节点数（下降沿）
    # print(f"[Stats] riseNodes={nL:.3f},fallNodes={nR:.3f}")

    # 位置 -> 该位置所有右侧节点索引（下降沿）列表
    fall_pos2idxs: Dict[int, List[int]] = defaultdict(list)
    for j, fpos in enumerate(fall_nodes):
        fall_pos2idxs[fpos].append(j)

    # 为了支持“负向时距”，还需要“位置 -> 上升沿索引列表”
    rise_pos2idxs: Dict[int, List[int]] = defaultdict(list)
    for iL, rpos in enumerate(rise_nodes):
        rise_pos2idxs[rpos].append(iL)

    # ---------------- 枚举候选边（左：上升沿索引 iL；右：下降沿索引 jR） ----------------
    # adjacency[iL] = [(jR, tag)]；tag标注类别：'L','C','CL','CC','LL','NC','NCL'
    adjacency: List[List[Tuple[int, str]]] = [[] for _ in range(nL)]

    # 把 target±tol 内的所有下降沿节点索引追加为候选
    def add_targets_for_length(iL: int, base_pos: int, length: int, tol: int, tag: str):
        # positive方向：falling 在 base_pos + length ± tol
        # negative方向：falling 在 base_pos - length ± tol（对 NEG_* 传入负的 length 即可在外层统一处理）
        # 这里传入的是实际目标位置 target = base_pos + length (+/- dt)
        for dt in range(-tol, tol + 1):
            tpos = base_pos + length + dt
            lst = fall_pos2idxs.get(tpos)
            if lst:
                # 把所有该位置的下降沿节点都连成边
                adjacency[iL].extend((j, tag) for j in lst)

    # 为每个上升沿列举正向/反向的所有候选边
    for iL, rpos in enumerate(rise_nodes):
        # 正向单段：lead / comp
        for L, tag, tol in POS_SINGLE:
            add_targets_for_length(iL, rpos, +L, tol, tag)
        # 正向复合：CL / CC / LL
        for L, tag, tol in POS_COMPOSITE:
            add_targets_for_length(iL, rpos, +L, tol, tag)
        # 反向单段：负comp（fall在前、rise在后） -> 等价为 rising 连接到更早的 falling
        for L, tag, tol in NEG_SINGLE:
            add_targets_for_length(iL, rpos, -L, tol, tag)
        # 反向复合：负(comp-lead)
        for L, tag, tol in NEG_COMPOSITE:
            add_targets_for_length(iL, rpos, -(L), tol, tag)

    # ---------------- Hopcroft–Karp 最大匹配（忽略边权，只最大数量） ----------------
    # 为了最小剩余事件数，本质就是最大基数匹配
    NIL = -1
    pairU = np.full(nL, NIL, dtype=np.int32)  # U->V 的匹配
    pairV = np.full(nR, NIL, dtype=np.int32)  # V->U 的匹配
    dist  = np.zeros(nL, dtype=np.int32)

    # 只保留邻接里的“索引”（忽略 tag），另存一份 tag 索引便于回溯生成脉冲
    adj_idx: List[List[int]] = [[] for _ in range(nL)]
    adj_tag: List[List[str]] = [[] for _ in range(nL)]
    for i in range(nL):
        if adjacency[i]:
            js, ts = zip(*adjacency[i])
            adj_idx[i] = list(js)
            adj_tag[i] = list(ts)

    def bfs() -> bool:
        q = deque()
        for u in range(nL):
            if pairU[u] == NIL:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = 1_000_000_000
        found_augment = False
        while q:
            u = q.popleft()
            for v in adj_idx[u]:
                pu = pairV[v]
                if pu == NIL:
                    found_augment = True
                else:
                    if dist[pu] == 1_000_000_000:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
        return found_augment

    def dfs(u: int) -> bool:
        for k, v in enumerate(adj_idx[u]):
            pu = pairV[v]
            if pu == NIL or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = 1_000_000_000
        return False

    # 主循环
    while bfs():
        for u in range(nL):
            if pairU[u] == NIL:
                dfs(u)

    # ---------------- 根据匹配生成脉冲 ----------------
    pulses: List[Pulse] = []
    # 为了取回每条匹配的 tag，需要从 u 的邻接里找到与 v 匹配的那条边
    # 注意：可能存在多个相同 v（不同 tag）的边；我们优先选择“单段”而非“复合”（可选的稳定化策略）
    TAG_PRI = {'C': 3, 'L': 2, 'NC': 3, 'CL': 1, 'CC': 1, 'LL': 1, 'NCL': 1, 'NCC': 1}  # 单段>复合
    for u, v in enumerate(pairU):
        if v == NIL:
            continue
        rpos = rise_nodes[u]
        fpos = fall_nodes[v]
        # 找到该(u,v)最“优”的tag
        chosen_tag = None
        best_pri = -1
        for jj, vv in enumerate(adj_idx[u]):
            if vv == v:
                tg = adj_tag[u][jj]
                pri = TAG_PRI.get(tg, 0)
                if pri > best_pri:
                    best_pri = pri
                    chosen_tag = tg

        if fpos >= rpos:
            # 正向
            diff = fpos - rpos
            if chosen_tag == 'L':  # 先导
                pulses.append(Pulse(rpos, rpos + lead_samps - 1, lead_samps, +1, amp_base))
            elif chosen_tag == 'C':  # 正比特
                pulses.append(Pulse(rpos, rpos + comp_samps - 1, comp_samps, +1, amp_base))
            elif chosen_tag == 'CL':  # comp + lead
                pulses.append(Pulse(rpos, rpos + comp_samps - 1, comp_samps, +1, amp_base))
                pulses.append(Pulse(rpos + comp_samps, rpos + comp_samps + lead_samps - 1, lead_samps, +1, amp_base))
            elif chosen_tag == 'CC':  # comp + comp
                pulses.append(Pulse(rpos, rpos + comp_samps - 1, comp_samps, +1, amp_base))
                pulses.append(Pulse(rpos + comp_samps, rpos + 2*comp_samps - 1, comp_samps, +1, amp_base))
            elif chosen_tag == 'LL':  # lead + lead
                pulses.append(Pulse(rpos, rpos + lead_samps - 1, lead_samps, +1, amp_base))
                pulses.append(Pulse(rpos + lead_samps, rpos + 2*lead_samps - 1, lead_samps, +1, amp_base))
            else:
                # 理论不会走到；保底：按就近匹配长度判断
                if abs(diff - lead_samps) <= lead_tol:
                    pulses.append(Pulse(rpos, rpos + lead_samps - 1, lead_samps, +1, amp_base))
                elif abs(diff - comp_samps) <= comp_tol:
                    pulses.append(Pulse(rpos, rpos + comp_samps - 1, comp_samps, +1, amp_base))
        else:
            # 反向（负脉冲类）
            diff = rpos - fpos
            if chosen_tag == 'NC':  # 负comp
                pulses.append(Pulse(fpos, fpos + comp_samps - 1, comp_samps, -1, amp_base))
            elif chosen_tag == 'NCL':  # 负(comp-lead)：负comp + 正lead（嵌套/接续）
                pulses.append(Pulse(fpos, fpos + comp_samps - 1, comp_samps, -1, amp_base))
                start_lead = rpos  # = fpos + (comp - lead)
                pulses.append(Pulse(start_lead, start_lead + lead_samps - 1, lead_samps, +1, amp_base))
            elif chosen_tag == 'NCC':  # 新增：负 CC -> 两段连续负 comp
                pulses.append(Pulse(fpos, fpos + comp_samps - 1, comp_samps, -1, amp_base))
                pulses.append(Pulse(fpos + comp_samps, fpos + 2 * comp_samps - 1, comp_samps, -1, amp_base))
            else:
                # 保底识别
                if abs(diff - comp_samps) <= comp_tol:
                    pulses.append(Pulse(fpos, fpos + comp_samps - 1, comp_samps, -1, amp_base))

    pulses.sort(key=lambda p: p.start)
    # ---------------- 统计未匹配剩余事件 ----------------
    used_r = np.zeros(nL, dtype=bool); used_r[pairU != NIL] = True
    used_f = np.zeros(nR, dtype=bool); used_f[pairV != NIL] = True
    rise_left: Dict[int,int] = defaultdict(int)
    fall_left: Dict[int,int] = defaultdict(int)
    for i, rpos in enumerate(rise_nodes):
        if not used_r[i]:
            rise_left[rpos] += 1
    for j, fpos in enumerate(fall_nodes):
        if not used_f[j]:
            fall_left[fpos] += 1

    return pulses, dict(rise_left), dict(fall_left)

# ---------------- 脉冲分组----------------
def group_bits_fast(bits_pulses: List[Pulse],
                    num_bits: int,
                    spacing_ideal: int,
                    spacing_tol: int) -> Tuple[List[List[int]], np.ndarray]:

    n = len(bits_pulses)
    used = np.zeros(n, dtype=bool)

    starts = np.array([p.start for p in bits_pulses], dtype=np.int64)
    start_map: Dict[int, List[int]] = {}
    for idx, s in enumerate(starts):
        start_map.setdefault(s, []).append(idx)

    bit_groups: List[List[int]] = []

    # 以起点升序作为组起点尝试
    order = np.argsort(starts)
    for i in order:
        if used[i]:
            continue

        s0 = starts[i]

        # ------- 第2个脉冲候选：目标 s0 + spacing_ideal ± tol -------
        target1 = s0 + spacing_ideal
        second_candidates: List[Tuple[int, int]] = []  # (idx, abs_err)

        for dt in range(-spacing_tol, spacing_tol + 1):
            cand_list = start_map.get(target1 + dt)
            if not cand_list:
                continue
            for j in cand_list:
                if j == i or used[j]:
                    continue
                s1 = starts[j]
                abs_err = abs((s1 - s0) - spacing_ideal)
                second_candidates.append((j, abs_err))

        # 如无第二脉冲候选，换下一个起点
        if not second_candidates:
            continue

        # 按与理想间距的误差排序，逐个尝试
        second_candidates.sort(key=lambda x: x[1])

        grouped = False
        for j1, _ in second_candidates:
            if used[j1]:
                continue

            s1 = starts[j1]
            d = s1 - s0
            if d <= 0:
                continue  # 观测间距必须为正

            group = [i, j1]
            ok = True

            for b in range(2, num_bits):
                target = s1 + (b - 1) * d
                matched_j = -1

                for dt in range(-spacing_tol, spacing_tol + 1):
                    cand_list = start_map.get(target + dt)
                    if not cand_list:
                        continue
                    for j in cand_list:
                        if (not used[j]) and (j not in group):
                            matched_j = j
                            break
                    if matched_j >= 0:
                        break

                if matched_j < 0:
                    ok = False
                    break
                group.append(matched_j)

            if ok and len(group) == num_bits:
                used[group] = True
                bit_groups.append(group)
                grouped = True
                break

    return bit_groups, used

# ---------------- 先导匹配 ----------------
def match_leads_min_diff(leads: List[Pulse],
                         lead_used: np.ndarray,
                         first_start: int,
                         last_end: int) -> List[Tuple[int,int,int,int]]:
    # 收集候选
    start_cands_idx = []
    start_cands_val = []  # dist1 = first_start - (lead_end)
    end_cands_idx = []
    end_cands_val = []    # dist2 = (lead_start) - last_end

    for i, L in enumerate(leads):
        if lead_used[i]:
            continue
        L_end = L.start + L.length - 1
        if L_end < first_start:
            start_cands_idx.append(i)
            start_cands_val.append(first_start - L_end)
        elif L.start > last_end:
            end_cands_idx.append(i)
            end_cands_val.append(L.start - last_end)

    if not start_cands_idx or not end_cands_idx:
        return []

    # 两数组均已按 leads 顺序自然有序
    A = np.array(start_cands_val, dtype=np.int64)  # 升序（因为 L_end 递增 => dist1 递减；但我们按遍历顺序也足够）
    B = np.array(end_cands_val, dtype=np.int64)    # 升序

    # 为确保 A/B 升序，重新排序（保险起见）
    a_order = np.argsort(A); A = A[a_order]; start_idx_sorted = [start_cands_idx[k] for k in a_order]
    b_order = np.argsort(B); B = B[b_order]; end_idx_sorted   = [end_cands_idx[k] for k in b_order]

    i = j = 0
    best = 1 << 60
    pairs: List[Tuple[int,int,int,int]] = []

    while i < len(A) and j < len(B):
        d = abs(int(A[i]) - int(B[j]))
        if d < best:
            best = d
            pairs = [(start_idx_sorted[i], end_idx_sorted[j], int(A[i]), int(B[j]))]
        elif d == best:
            pairs.append((start_idx_sorted[i], end_idx_sorted[j], int(A[i]), int(B[j])))
        # 移动较小的一侧
        if A[i] < B[j]:
            i += 1
        elif A[i] > B[j]:
            j += 1
        else:
            # 完全相等，双向推进
            i += 1; j += 1

    return pairs

# ========= 热图 =========
def best_grid(n: int) -> Tuple[int, int]:
    r = int(np.floor(np.sqrt(n)))
    while r > 1 and (n % r != 0):
        r -= 1
    if r <= 1:
        return 1, n
    return r, n // r

def rc_to_dim(row: int, col: int, cols: int) -> int:
    return row * cols + col + 1  # 1-based

def int_to_bits(val: int, num_bits: int) -> np.ndarray:
    return ((val >> np.arange(num_bits - 1, -1, -1)) & 1).astype(np.int8)

def _bits_to_int(bits: np.ndarray) -> int:
    w = (1 << np.arange(bits.size-1, -1, -1, dtype=np.int64))
    return int(np.dot(bits.astype(np.int64), w))

def build_sensor_heatmaps(bits_cell: List[np.ndarray],
                          dim_array: np.ndarray,
                          subset_idx: np.ndarray,
                          decoded_info: List[DecodeInfo],
                          num_bits: int,
                          n_dim: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int,int]]:
    rows, cols = best_grid(n_dim)

    # 没有信号=0
    sim_1d = np.zeros(n_dim, dtype=np.float32)
    dec_1d = np.zeros(n_dim, dtype=np.float32)

    # 模拟/真实
    for node_idx in subset_idx:
        dimI = int(dim_array[node_idx])  # 1-based
        pos = dimI - 1                   # 0-based
        val = _bits_to_int(bits_cell[node_idx])
        sim_1d[pos] = val

    # 解码
    for rec in decoded_info:
        pos = int(rec.dimI) - 1
        if 0 <= pos < n_dim:
            dec_1d[pos] = _bits_to_int(rec.bits)

    sim = sim_1d.reshape(rows, cols)
    dec = dec_1d.reshape(rows, cols)

    # 绝对差（两侧至少一侧为 0 也会正常显示差异）
    diff = np.abs(sim - dec).astype(sim.dtype, copy=False)

    return sim, dec, diff, (rows, cols)

def plot_sensor_heatmaps(sim: np.ndarray,
                         dec: np.ndarray,
                         diff: np.ndarray,
                         num_bits: int,
                         title_prefix: str = "Sensor-array heatmaps"):
    vmax = float((1 << num_bits) - 1)  # 0..1023

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(sim, vmin=0, vmax=vmax, cmap='inferno', origin='upper', aspect='auto')
    axes[0].set_title("Simulated (ground truth)")
    axes[0].set_xlabel("cols"); axes[0].set_ylabel("rows")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Pressure")

    im1 = axes[1].imshow(dec, vmin=0, vmax=vmax, cmap='inferno', origin='upper', aspect='auto')
    axes[1].set_title("Decoded")
    axes[1].set_xlabel("cols"); axes[1].set_ylabel("rows")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Pressure")

    im2 = axes[2].imshow(diff, vmin=0, vmax=vmax, cmap='inferno', origin='upper', aspect='auto')
    axes[2].set_title("Abs difference")
    axes[2].set_xlabel("cols"); axes[2].set_ylabel("rows")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="|error|")

    #fig.suptitle(title_prefix, y=1.02, fontsize=12)
    plt.show()

def chaikin_smooth_closed(ctrl_pts: np.ndarray, rounds: int = 2) -> np.ndarray:
    """
    Chaikin 角切算法将闭合折线平滑成近似样条。输入/输出坐标均为 (N,2) 归一化点，闭合多边形默认首尾连线。
    """
    pts = np.asarray(ctrl_pts, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 2
    for _ in range(max(0, rounds)):
        P = []
        # 首尾相连
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i+1) % len(pts)]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            P.extend([Q, R])
        pts = np.asarray(P, dtype=np.float32)
    return pts

def place_image_mask(rows, cols, center, radius, angle_deg, img_mask01):
    """
    将 0..1 的 img_mask01（H x W）按中心/缩放/旋转“贴”到 rows x cols 的网格上。
    radius 是归一化的“半径”标度（以宽度为基准），等效控制贴图的外接圆尺寸。
    """
    cy, cx = center
    Ih, Iw = img_mask01.shape
    # 先旋转
    ang = np.deg2rad(angle_deg)
    # 构造目标网格坐标
    yy, xx = np.meshgrid(np.linspace(0, 1, rows, endpoint=False),
                         np.linspace(0, 1, cols, endpoint=False), indexing='ij')
    # 将全局坐标映射到“贴图局部坐标”[-1,1]（先平移，再按 radius 缩放，再逆旋转）
    x0 = xx - cx
    y0 = yy - cy
    xr =  x0 * np.cos(-ang) + y0 * np.sin(-ang)
    yr = -x0 * np.sin(-ang) + y0 * np.cos(-ang)

    # 设定使得 radius 覆盖到局部坐标的单位圆：|x|,|y|<=radius -> 归一化到 [-1,1]
    # 这里按宽度标定，保持原图纵横比
    u = xr / (radius + 1e-12)
    v = yr / (radius + 1e-12)
    # 将 [-1,1] 映射到原图像素坐标
    iu = (u * 0.5 + 0.5) * (Iw - 1)
    iv = (v * 0.5 + 0.5) * (Ih - 1)

    # 双线性采样（向量化）
    iu0 = np.floor(iu).astype(int).clip(0, Iw-1)
    iv0 = np.floor(iv).astype(int).clip(0, Ih-1)
    iu1 = np.ceil(iu).astype(int).clip(0, Iw-1)
    iv1 = np.ceil(iv).astype(int).clip(0, Ih-1)
    du = iu - iu0
    dv = iv - iv0
    m00 = img_mask01[iv0, iu0]
    m01 = img_mask01[iv1, iu0]
    m10 = img_mask01[iv0, iu1]
    m11 = img_mask01[iv1, iu1]
    mask = (m00*(1-du)*(1-dv) + m10*du*(1-dv) + m01*(1-du)*dv + m11*du*dv)

    # 超出贴图范围的（|u|>1 或 |v|>1）强制为 0
    outside = (np.abs(u) > 1) | (np.abs(v) > 1)
    mask[outside] = 0.0
    return mask.astype(np.float32)

def make_pattern(rows: int,
                 cols: int,
                 kind: str = "circle",
                 center=None,             # (cy,cx) ∈ [0,1]
                 radius: float = 0.3,
                 r_inner: float = 0.15,
                 sigma: float = 0.12,
                 angle_deg: float = 0.0,
                 aspect: float = 1.0,
                 # === 新增可选参数（不破坏旧用法）===
                 verts=None,              # "polygon"：[(x,y), ...] 归一化
                 ctrl_pts=None,           # "spline_polygon"：控制点（闭合），会平滑成多边形
                 smooth_rounds: int = 2,
                 img_mask01: np.ndarray = None,  # "image"：0..1 灰度/二值图（H x W）
                 img_threshold: float = 0.5,     # "image"：二值化阈值
                 invert: bool = False            # "image"：是否反相（蓝色->白/黑 取决于你的素材）
                 ) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(0, 1, rows, endpoint=False),
                         np.linspace(0, 1, cols, endpoint=False), indexing='ij')
    if center is None:
        cy, cx = 0.5, 0.5
    else:
        cy, cx = center

    y0 = yy - cy
    x0 = xx - cx

    th = np.deg2rad(angle_deg)
    xr =  x0 * np.cos(th) + y0 * np.sin(th)
    yr = -x0 * np.sin(th) + y0 * np.cos(th)

    out = np.zeros((rows, cols), dtype=np.float32)

    k = kind.lower()
    if k == "circle":
        r = np.sqrt(x0**2 + y0**2)
        out = (r <= radius).astype(np.float32)

    elif k == "ring":
        r = np.sqrt(x0**2 + y0**2)
        out = ((r >= r_inner) & (r <= radius)).astype(np.float32)

    elif k == "gaussian_finger":
        sigx = max(1e-6, sigma)
        sigy = max(1e-6, sigma * aspect)
        g = np.exp(-0.5 * ((xr / sigx)**2 + (yr / sigy)**2))
        g /= (g.max() + 1e-12)
        out = g.astype(np.float32)

    elif k == "ellipse":
        a = max(1e-6, radius)
        b = max(1e-6, radius * aspect)
        val = (xr / a)**2 + (yr / b)**2
        out = (val <= 1.0).astype(np.float32)

    elif k == "square":
        half_w = radius
        half_h = radius * aspect
        out = ((np.abs(xr) <= half_w) & (np.abs(yr) <= half_h)).astype(np.float32)

    elif k == "zju_star":
        R  = radius
        r0 = max(1e-6, r_inner) * R
        base = np.deg2rad(-90)
        verts_star = []
        for i in range(10):
            ang = base + th + i * (np.pi / 5.0)
            rad = R if (i % 2 == 0) else r0
            vx = cx + rad * np.cos(ang)
            vy = cy + rad * np.sin(ang)
            verts_star.append((vx, vy))
        verts_star = np.asarray(verts_star, dtype=np.float32)
        out = polygon_mask(rows, cols, verts_star)

    elif k == "polygon":
        assert verts is not None and len(verts) >= 3, "polygon 需要 verts=[(x,y),...]"
        V = np.asarray(verts, dtype=np.float32)
        out = polygon_mask(rows, cols, V)

    elif k == "spline_polygon":
        assert ctrl_pts is not None and len(ctrl_pts) >= 3, "spline_polygon 需要 ctrl_pts"
        V = chaikin_smooth_closed(np.asarray(ctrl_pts, dtype=np.float32),
                                  rounds=smooth_rounds)
        out = polygon_mask(rows, cols, V)

    elif k == "image":
        # img_mask01 是 0..1 的灰度或二值图（H x W）；>img_threshold 视为“内部”
        assert img_mask01 is not None and img_mask01.ndim == 2, \
            "image 需要 img_mask01 为二维 0..1 数组（灰度/二值）"
        m = img_mask01.astype(np.float32)
        if invert:
            m = 1.0 - m
        # 将灰度阈值化为 0/1，再轻微归一化
        m = (m > float(img_threshold)).astype(np.float32)
        # 贴图到阵列
        out = place_image_mask(rows, cols, (cy, cx), radius, angle_deg, m)

    else:
        raise ValueError(f"Unknown pattern kind: {kind}")

    if out.max() > 0:
        out = (out - out.min()) / (out.max() - out.min() + 1e-12)
    return out.astype(np.float32)

def polygon_mask(rows: int, cols: int, verts_xy: np.ndarray) -> np.ndarray:
    """
    点在多边形内测试（ray casting），返回 0/1 mask，verts_xy 形状 (M,2)，坐标在归一化 [0,1] 系里。
    向量化实现，阵列规模 ~1e3 量级很快。
    """
    yy, xx = np.meshgrid(np.linspace(0, 1, rows, endpoint=False),
                         np.linspace(0, 1, cols, endpoint=False), indexing='ij')
    x = xx.ravel()
    y = yy.ravel()
    px = verts_xy[:, 0]; py = verts_xy[:, 1]
    # 关闭多边形
    x1 = px;             y1 = py
    x2 = np.roll(px, -1); y2 = np.roll(py, -1)

    # 计算每条边的相交布尔值
    # ((y1>y)!=(y2>y)) & (x < x_intersect)
    # x_intersect = x1 + (y - y1)*(x2 - x1)/(y2 - y1)
    eps = 1e-12
    y = y[:, None]
    x = x[:, None]
    cond = ((y1 > y) != (y2 > y))
    x_inter = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) + eps)
    hit = cond & (x < x_inter)
    inside = np.mod(np.count_nonzero(hit, axis=1), 2).astype(np.float32)
    return inside.reshape(rows, cols)

# 5) 从图案中挑选 num_nodes 个激活位置，并量化为 0..(2^num_bits-1)
def assign_nodes_from_pattern(pattern: np.ndarray,
                              num_nodes: int,
                              num_bits: int,
                              pick_mode: str = "topk"   # "topk" | "weighted"
                              ):
    rows, cols = pattern.shape
    flat = pattern.ravel()
    vmax = (1 << num_bits) - 1

    if pick_mode == "topk":
        # 选强度最大的 k 个位置（若 k > 正值数目，自动补零强度位置）
        idx_sorted = np.argsort(flat)[::-1]
        idx_pick = idx_sorted[:num_nodes]
    elif pick_mode == "weighted":
        w = flat.clip(min=0)
        if w.sum() <= 0:
            # 全零就随机
            idx_pick = np.random.choice(flat.size, size=num_nodes, replace=False)
        else:
            w = w / w.sum()
            idx_pick = np.random.choice(flat.size, size=num_nodes, replace=False, p=w)
    else:
        raise ValueError("pick_mode must be 'topk' or 'weighted'")

    # 被选位置的十进制“强度值”，线性量化到 0..vmax
    vals01 = flat[idx_pick]
    vals_q = np.rint(vals01 * vmax).astype(np.int32)  # 0..vmax

    # 计算这些位置的 (row, col) 与 dimI（1-based）
    rows_idx = idx_pick // cols
    cols_idx = idx_pick %  cols
    dims_1based = rows_idx * cols + cols_idx + 1

    # 生成 bits（高位在前）
    bits_list = [int_to_bits(int(v), num_bits) for v in vals_q]

    return dims_1based.astype(int), bits_list, (rows_idx, cols_idx), vals_q

# ========= 噪声添加 =========
def signal_power(x: np.ndarray) -> float:
    """返回实值信号的平均功率 E[x^2]。"""
    if x.size == 0:
        return 0.0
    # 用 float64 计算减少数值误差
    return float(np.mean(x.astype(np.float64)**2))

def add_awgn(
    x: np.ndarray,
    snr_db: Optional[float] = None,
    snr_linear: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    dtype=np.float32
) -> Tuple[np.ndarray, float, float]:
    """
    按给定 SNR 向实值信号 x 添加高斯白噪声（AWGN）。
    - snr_db 或 snr_linear 必须给一个（另一个留空）。
    - 返回: (y_noisy, actual_snr_db, noise_sigma)
    """
    assert (snr_db is not None) ^ (snr_linear is not None), "snr_db 与 snr_linear 二选一"
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x)
    Ps = signal_power(x)
    if Ps <= 0:
        # 全零或近零信号：以单位方差作为基准加噪（或直接返回噪声）
        sigma = 1.0
        noise = rng.normal(0.0, sigma, size=x.shape)
        y = (x.astype(np.float64) + noise).astype(dtype)
        # 实测 SNR 无意义（Ps≈0）
        return y, float("-inf"), sigma

    SNR_lin = snr_linear if snr_linear is not None else (10.0 ** (snr_db / 10.0))
    Pn = Ps / SNR_lin
    sigma = np.sqrt(Pn)

    noise = rng.normal(0.0, sigma, size=x.shape)
    y = (x.astype(np.float64) + noise).astype(dtype)

    # 计算实测 SNR（考虑一次随机实现的偏差）
    Pn_meas = signal_power(y.astype(np.float64) - x.astype(np.float64))
    Ps_meas = signal_power(x.astype(np.float64))
    actual_snr_lin = Ps_meas / max(Pn_meas, 1e-30)
    actual_snr_db = 10.0 * np.log10(actual_snr_lin)
    return y, actual_snr_db, sigma

if __name__ == "__main__":
    main()