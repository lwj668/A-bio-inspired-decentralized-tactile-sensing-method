from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Optional, List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

@dataclass
class Pulse:
    start: int
    end: int
    length: int
    amp_sign: int
    peak_val: float

@dataclass
class DecodeInfo:
    node_id: int
    dimI: int
    bits: np.ndarray
    startLeadP: Pulse
    endLeadP: Pulse
    bitPulses: List[Pulse]

def main():
    fs        = 200e6
    amp       = 0.4
    num_bits  = 10
    num_nodes = 500
    n_dim     = 3000

    T_lead_us  = 0.05
    lead_samps = int(round(T_lead_us * 1e-6 * fs))
    T_comp_us  = 0.5
    comp_samps = int(round(T_comp_us * 1e-6 * fs))

    comp_tol = max(0, int(round(0.001 * comp_samps)))
    lead_tol = max(0, int(round(0.0001 * lead_samps)))

    spacing_ideal = n_dim * comp_samps
    spacing_tol   = int(round(0.005 * comp_samps))

    max_offset_us    = 5000
    max_offset_samps = int(round(max_offset_us * 1e-6 * fs))

    rng = np.random.default_rng()
    if num_nodes > n_dim:
        raise ValueError("节点数量不能大于维度")

    rows, cols = best_grid(n_dim)

    img_path = "浙大校标.png"
    img = Image.open(img_path).convert("L")
    img = np.asarray(img, dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)

    pattern_logo = make_pattern(
        rows=rows, cols=cols,
        kind="image",
        center=(0.5, 0.5),
        radius=0.38,
        angle_deg=0,
        img_mask01=img,
        img_threshold=0.70,
        invert=True
    )

    pattern_soft = pattern_logo * make_pattern(
        rows=rows, cols=cols,
        kind="gaussian_finger",
        center=(0.5, 0.5),
        sigma=0.22,
        aspect=1.1
    )
    pattern_soft = (pattern_soft - pattern_soft.min()) / (pattern_soft.max() - pattern_soft.min() + 1e-12)

    dims_sel, bits_list, (r_idx, c_idx), vals_q = assign_nodes_from_pattern(
        pattern_soft, num_nodes, num_bits, pick_mode="topk"
    )

    all_dims = np.arange(1, n_dim + 1, dtype=int)

    mask_sel = np.zeros_like(all_dims, dtype=bool)
    mask_sel[dims_sel - 1] = True
    rest_dims = all_dims[~mask_sel]
    rng.shuffle(rest_dims)
    dim_array = np.concatenate([dims_sel, rest_dims])

    bits_cell: List[np.ndarray] = [None] * num_nodes
    for i in range(num_nodes):
        bits_cell[i] = bits_list[i]

    frame_cell: List[np.ndarray] = []
    frame_len = np.zeros(num_nodes, dtype=np.int64)
    for i in range(num_nodes):
        dimI = int(dim_array[i])
        w = build_one_node_frame_updated(bits_cell[i], dimI, n_dim, comp_samps, lead_samps, amp)
        frame_cell.append(w)
        frame_len[i] = w.size

    big_len = int(frame_len.max()) + max_offset_samps + 1000
    wave_sub = np.zeros(big_len, dtype=np.float32)
    subset = np.arange(num_nodes, dtype=int)
    offsets = rng.integers(0, max_offset_samps + 1, size=num_nodes, dtype=np.int64)

    for idx in subset:
        w = frame_cell[idx]
        off = int(offsets[idx])
        wave_sub[off:off+w.size] += w

    T0 = time.perf_counter()
    decoded, pulses, leads, bitsPulses, bitGroups, stats, _rise_left, _fall_left = decode_all_nodes_fast_v2(
        wave_sub, fs, amp, num_bits, n_dim, comp_samps, lead_samps, comp_tol, lead_tol, spacing_ideal, spacing_tol
    )
    T1 = time.perf_counter()

    print(f"[Stats] events={stats['t_events']:.3f}s, group={stats['t_group']:.3f}s, leads={stats['t_leads']:.3f}s, total={T1-T0:.3f}s")
    print(f"[Counts] pulses={stats['n_pulses']}, leads={stats['n_leads']}, bits={stats['n_bits']}, groups={stats['n_groups']}, decoded={stats['n_decoded']}")

    diff = np.zeros(wave_sub.size + 1, dtype=np.float32)
    for rec in decoded:
        for p in [rec.startLeadP] + rec.bitPulses + [rec.endLeadP]:
            val = p.amp_sign * amp
            diff[p.start] += val
            diff[p.end + 1] -= val
    wave_recon = np.cumsum(diff[:-1])
    wave_resid = wave_sub - wave_recon

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

def build_one_node_frame_updated(bits: np.ndarray,
                                 dimI: int,
                                 n_dim: int,
                                 comp_samps: int,
                                 lead_samps: int,
                                 amp: float) -> np.ndarray:
    num_bits = bits.size
    if num_bits != 10:
        raise ValueError("假定每节点10比特。")
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

    t0 = time.perf_counter()
    pulses, _rise_left, _fall_left = detect_pulses_multiset(wave, amp, comp_samps, lead_samps, comp_tol, lead_tol)
    t1 = time.perf_counter()

    lead_tol = int(round(0.1 * lead_samps))
    leads: List[Pulse] = []
    bits:  List[Pulse] = []
    for p in pulses:
        if (p.amp_sign > 0) and (abs(p.length - lead_samps) <= lead_tol):
            leads.append(p)
        else:
            bits.append(p)

    bits.sort(key=lambda p: p.start)
    bit_groups, used_bits = group_bits_fast(bits, num_bits, spacing_ideal, spacing_tol)
    t2 = time.perf_counter()

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

def detect_pulses_multiset(signal: np.ndarray,
                           amp_base: float,
                           comp_samps: int,
                           lead_samps: int,
                           comp_tol: int,
                           lead_tol: int,
                           ) -> Tuple[List[Pulse], Dict[int,int], Dict[int,int]]:

    total_tol = max(comp_tol, lead_tol)

    POS_SINGLE = [(lead_samps, 'L', lead_tol),
                  (comp_samps, 'C', comp_tol)]
    POS_COMPOSITE = [(comp_samps + lead_samps, 'CL', total_tol),
                     (2 * comp_samps, 'CC', comp_tol),
                     (2 * lead_samps, 'LL', lead_tol)]

    NEG_SINGLE = [(comp_samps, 'NC', comp_tol)]
    NEG_COMPOSITE = [(comp_samps - lead_samps, 'NCL', total_tol),
                     (2 * comp_samps, 'NCC', comp_tol)]

    c = np.rint(signal / amp_base).astype(np.int16)
    delta = np.diff(c)
    idx = np.nonzero(delta != 0)[0]

    rise_nodes: List[int] = []
    fall_nodes: List[int] = []
    for i in idx:
        pos = i + 1
        dv = int(delta[i])
        if dv > 0:
            rise_nodes.extend([pos] * dv)
        else:
            fall_nodes.extend([pos] * (-dv))

    nL = len(rise_nodes)
    nR = len(fall_nodes)

    fall_pos2idxs: Dict[int, List[int]] = defaultdict(list)
    for j, fpos in enumerate(fall_nodes):
        fall_pos2idxs[fpos].append(j)

    rise_pos2idxs: Dict[int, List[int]] = defaultdict(list)
    for iL, rpos in enumerate(rise_nodes):
        rise_pos2idxs[rpos].append(iL)

    adjacency: List[List[Tuple[int, str]]] = [[] for _ in range(nL)]

    def add_targets_for_length(iL: int, base_pos: int, length: int, tol: int, tag: str):

        for dt in range(-tol, tol + 1):
            tpos = base_pos + length + dt
            lst = fall_pos2idxs.get(tpos)
            if lst:

                adjacency[iL].extend((j, tag) for j in lst)

    for iL, rpos in enumerate(rise_nodes):

        for L, tag, tol in POS_SINGLE:
            add_targets_for_length(iL, rpos, +L, tol, tag)

        for L, tag, tol in POS_COMPOSITE:
            add_targets_for_length(iL, rpos, +L, tol, tag)

        for L, tag, tol in NEG_SINGLE:
            add_targets_for_length(iL, rpos, -L, tol, tag)

        for L, tag, tol in NEG_COMPOSITE:
            add_targets_for_length(iL, rpos, -(L), tol, tag)

    NIL = -1
    pairU = np.full(nL, NIL, dtype=np.int32)
    pairV = np.full(nR, NIL, dtype=np.int32)
    dist  = np.zeros(nL, dtype=np.int32)

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

    while bfs():
        for u in range(nL):
            if pairU[u] == NIL:
                dfs(u)

    pulses: List[Pulse] = []

    TAG_PRI = {'C': 3, 'L': 2, 'NC': 3, 'CL': 1, 'CC': 1, 'LL': 1, 'NCL': 1, 'NCC': 1}
    for u, v in enumerate(pairU):
        if v == NIL:
            continue
        rpos = rise_nodes[u]
        fpos = fall_nodes[v]

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

            diff = fpos - rpos
            if chosen_tag == 'L':
                pulses.append(Pulse(rpos, rpos + lead_samps - 1, lead_samps, +1, amp_base))
            elif chosen_tag == 'C':
                pulses.append(Pulse(rpos, rpos + comp_samps - 1, comp_samps, +1, amp_base))
            elif chosen_tag == 'CL':
                pulses.append(Pulse(rpos, rpos + comp_samps - 1, comp_samps, +1, amp_base))
                pulses.append(Pulse(rpos + comp_samps, rpos + comp_samps + lead_samps - 1, lead_samps, +1, amp_base))
            elif chosen_tag == 'CC':
                pulses.append(Pulse(rpos, rpos + comp_samps - 1, comp_samps, +1, amp_base))
                pulses.append(Pulse(rpos + comp_samps, rpos + 2*comp_samps - 1, comp_samps, +1, amp_base))
            elif chosen_tag == 'LL':
                pulses.append(Pulse(rpos, rpos + lead_samps - 1, lead_samps, +1, amp_base))
                pulses.append(Pulse(rpos + lead_samps, rpos + 2*lead_samps - 1, lead_samps, +1, amp_base))
            else:

                if abs(diff - lead_samps) <= lead_tol:
                    pulses.append(Pulse(rpos, rpos + lead_samps - 1, lead_samps, +1, amp_base))
                elif abs(diff - comp_samps) <= comp_tol:
                    pulses.append(Pulse(rpos, rpos + comp_samps - 1, comp_samps, +1, amp_base))
        else:

            diff = rpos - fpos
            if chosen_tag == 'NC':
                pulses.append(Pulse(fpos, fpos + comp_samps - 1, comp_samps, -1, amp_base))
            elif chosen_tag == 'NCL':
                pulses.append(Pulse(fpos, fpos + comp_samps - 1, comp_samps, -1, amp_base))
                start_lead = rpos
                pulses.append(Pulse(start_lead, start_lead + lead_samps - 1, lead_samps, +1, amp_base))
            elif chosen_tag == 'NCC':
                pulses.append(Pulse(fpos, fpos + comp_samps - 1, comp_samps, -1, amp_base))
                pulses.append(Pulse(fpos + comp_samps, fpos + 2 * comp_samps - 1, comp_samps, -1, amp_base))
            else:

                if abs(diff - comp_samps) <= comp_tol:
                    pulses.append(Pulse(fpos, fpos + comp_samps - 1, comp_samps, -1, amp_base))

    pulses.sort(key=lambda p: p.start)

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

    order = np.argsort(starts)
    for i in order:
        if used[i]:
            continue

        s0 = starts[i]

        target1 = s0 + spacing_ideal
        second_candidates: List[Tuple[int, int]] = []

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

        if not second_candidates:
            continue

        second_candidates.sort(key=lambda x: x[1])

        grouped = False
        for j1, _ in second_candidates:
            if used[j1]:
                continue

            s1 = starts[j1]
            d = s1 - s0
            if d <= 0:
                continue

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

def match_leads_min_diff(leads: List[Pulse],
                         lead_used: np.ndarray,
                         first_start: int,
                         last_end: int) -> List[Tuple[int,int,int,int]]:

    start_cands_idx = []
    start_cands_val = []
    end_cands_idx = []
    end_cands_val = []

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

    A = np.array(start_cands_val, dtype=np.int64)
    B = np.array(end_cands_val, dtype=np.int64)

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

        if A[i] < B[j]:
            i += 1
        elif A[i] > B[j]:
            j += 1
        else:

            i += 1; j += 1

    return pairs

def best_grid(n: int) -> Tuple[int, int]:
    r = int(np.floor(np.sqrt(n)))
    while r > 1 and (n % r != 0):
        r -= 1
    if r <= 1:
        return 1, n
    return r, n // r

def rc_to_dim(row: int, col: int, cols: int) -> int:
    return row * cols + col + 1

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

    sim_1d = np.zeros(n_dim, dtype=np.float32)
    dec_1d = np.zeros(n_dim, dtype=np.float32)

    for node_idx in subset_idx:
        dimI = int(dim_array[node_idx])
        pos = dimI - 1
        val = _bits_to_int(bits_cell[node_idx])
        sim_1d[pos] = val

    for rec in decoded_info:
        pos = int(rec.dimI) - 1
        if 0 <= pos < n_dim:
            dec_1d[pos] = _bits_to_int(rec.bits)

    sim = sim_1d.reshape(rows, cols)
    dec = dec_1d.reshape(rows, cols)

    diff = np.abs(sim - dec).astype(sim.dtype, copy=False)

    return sim, dec, diff, (rows, cols)

def plot_sensor_heatmaps(sim: np.ndarray,
                         dec: np.ndarray,
                         diff: np.ndarray,
                         num_bits: int,
                         title_prefix: str = "Sensor-array heatmaps"):
    vmax = float((1 << num_bits) - 1)

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

    plt.show()

def chaikin_smooth_closed(ctrl_pts: np.ndarray, rounds: int = 2) -> np.ndarray:
    pts = np.asarray(ctrl_pts, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 2
    for _ in range(max(0, rounds)):
        P = []

        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i+1) % len(pts)]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            P.extend([Q, R])
        pts = np.asarray(P, dtype=np.float32)
    return pts

def place_image_mask(rows, cols, center, radius, angle_deg, img_mask01):
    cy, cx = center
    Ih, Iw = img_mask01.shape

    ang = np.deg2rad(angle_deg)

    yy, xx = np.meshgrid(np.linspace(0, 1, rows, endpoint=False),
                         np.linspace(0, 1, cols, endpoint=False), indexing='ij')

    x0 = xx - cx
    y0 = yy - cy
    xr =  x0 * np.cos(-ang) + y0 * np.sin(-ang)
    yr = -x0 * np.sin(-ang) + y0 * np.cos(-ang)

    u = xr / (radius + 1e-12)
    v = yr / (radius + 1e-12)

    iu = (u * 0.5 + 0.5) * (Iw - 1)
    iv = (v * 0.5 + 0.5) * (Ih - 1)

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

    outside = (np.abs(u) > 1) | (np.abs(v) > 1)
    mask[outside] = 0.0
    return mask.astype(np.float32)

def make_pattern(rows: int,
                 cols: int,
                 kind: str = "circle",
                 center=None,
                 radius: float = 0.3,
                 r_inner: float = 0.15,
                 sigma: float = 0.12,
                 angle_deg: float = 0.0,
                 aspect: float = 1.0,

                 verts=None,
                 ctrl_pts=None,
                 smooth_rounds: int = 2,
                 img_mask01: np.ndarray = None,
                 img_threshold: float = 0.5,
                 invert: bool = False
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

        assert img_mask01 is not None and img_mask01.ndim == 2,            "image 需要 img_mask01 为二维数组"
        m = img_mask01.astype(np.float32)
        if invert:
            m = 1.0 - m

        m = (m > float(img_threshold)).astype(np.float32)

        out = place_image_mask(rows, cols, (cy, cx), radius, angle_deg, m)

    else:
        raise ValueError(f"Unknown pattern kind: {kind}")

    if out.max() > 0:
        out = (out - out.min()) / (out.max() - out.min() + 1e-12)
    return out.astype(np.float32)

def polygon_mask(rows: int, cols: int, verts_xy: np.ndarray) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(0, 1, rows, endpoint=False),
                         np.linspace(0, 1, cols, endpoint=False), indexing='ij')
    x = xx.ravel()
    y = yy.ravel()
    px = verts_xy[:, 0]; py = verts_xy[:, 1]

    x1 = px;             y1 = py
    x2 = np.roll(px, -1); y2 = np.roll(py, -1)

    eps = 1e-12
    y = y[:, None]
    x = x[:, None]
    cond = ((y1 > y) != (y2 > y))
    x_inter = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) + eps)
    hit = cond & (x < x_inter)
    inside = np.mod(np.count_nonzero(hit, axis=1), 2).astype(np.float32)
    return inside.reshape(rows, cols)

def assign_nodes_from_pattern(pattern: np.ndarray,
                              num_nodes: int,
                              num_bits: int,
                              pick_mode: str = "topk"
                              ):
    rows, cols = pattern.shape
    flat = pattern.ravel()
    vmax = (1 << num_bits) - 1

    if pick_mode == "topk":

        idx_sorted = np.argsort(flat)[::-1]
        idx_pick = idx_sorted[:num_nodes]
    elif pick_mode == "weighted":
        w = flat.clip(min=0)
        if w.sum() <= 0:

            idx_pick = np.random.choice(flat.size, size=num_nodes, replace=False)
        else:
            w = w / w.sum()
            idx_pick = np.random.choice(flat.size, size=num_nodes, replace=False, p=w)
    else:
        raise ValueError("pick_mode must be 'topk' or 'weighted'")

    vals01 = flat[idx_pick]
    vals_q = np.rint(vals01 * vmax).astype(np.int32)

    rows_idx = idx_pick // cols
    cols_idx = idx_pick %  cols
    dims_1based = rows_idx * cols + cols_idx + 1

    bits_list = [int_to_bits(int(v), num_bits) for v in vals_q]

    return dims_1based.astype(int), bits_list, (rows_idx, cols_idx), vals_q

def signal_power(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0

    return float(np.mean(x.astype(np.float64)**2))

def add_awgn(
    x: np.ndarray,
    snr_db: Optional[float] = None,
    snr_linear: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    dtype=np.float32
) -> Tuple[np.ndarray, float, float]:
    assert (snr_db is not None) ^ (snr_linear is not None), "snr_db 与 snr_linear 二选一"
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x)
    Ps = signal_power(x)
    if Ps <= 0:

        sigma = 1.0
        noise = rng.normal(0.0, sigma, size=x.shape)
        y = (x.astype(np.float64) + noise).astype(dtype)

        return y, float("-inf"), sigma

    SNR_lin = snr_linear if snr_linear is not None else (10.0 ** (snr_db / 10.0))
    Pn = Ps / SNR_lin
    sigma = np.sqrt(Pn)

    noise = rng.normal(0.0, sigma, size=x.shape)
    y = (x.astype(np.float64) + noise).astype(dtype)

    Pn_meas = signal_power(y.astype(np.float64) - x.astype(np.float64))
    Ps_meas = signal_power(x.astype(np.float64))
    actual_snr_lin = Ps_meas / max(Pn_meas, 1e-30)
    actual_snr_db = 10.0 * np.log10(actual_snr_lin)
    return y, actual_snr_db, sigma

if __name__ == "__main__":
    main()