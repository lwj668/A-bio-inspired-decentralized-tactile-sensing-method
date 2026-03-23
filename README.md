# Bio-Inspired Decentralized Tactile Sensing for Scalable Large-Area E-Skin

## Overview

Tactile perception in biological systems is not a passive process of raw data collection, but a highly efficient filtering and encoding mechanism performed by the nervous system. Humans can effortlessly recognize complex objects and textures through distributed afferent nerves that preprocess external stimuli into sparse, high-salience neural spikes. In contrast, most robotic tactile systems remain constrained by von Neumann-style architectures, where every touch event is treated as a redundant stream of data. This greatly limits scalability, transmission efficiency, and temporal performance, making it difficult to approach the sensing capability of human skin.

In this work, we propose a **bio-inspired decentralized tactile sensing framework** for scalable large-area electronic skin (e-skin). Building on an **orthogonal digital encoding strategy**, each tactile receptor generates a unique pulse train. The magnitude of tactile force is encoded into **spatiotemporal pulse patterns**, while the tactile distribution information is represented by **first-spike timing**. To decode these signals, we further develop a **computationally efficient spiking decoding algorithm** based on pulse grouping and decomposition, achieving a sensing latency of less than **6–8 ms**.

This framework enables a decentralized and biologically inspired mode of signal transmission and processing, where signals from individual tactile units are conveyed independently with extremely low mutual interference. The system maintains an error rate of only **0.01** even when **thousands of tactile units** are triggered simultaneously. In addition, we design a **highly scalable flexible modular tactile receptor** capable of pressure sensing and encoded signal generation, which can be conveniently assembled for large-area integration. Finally, nearly **100 modular tactile receptors** are integrated onto a humanoid robotic platform. Experimental results demonstrate robust **multi-point tactile sensing** and **object shape recognition**, highlighting the strong potential of this approach for next-generation intelligent humanoid robots.

---

## Key Contributions

- **Bio-inspired decentralized tactile sensing architecture** for scalable large-area e-skin
- **Orthogonal digital encoding strategy** for independent signal transmission with minimal interference
- **Spatiotemporal pulse-based tactile representation**, encoding both force magnitude and spatial distribution
- **Efficient spike decoding algorithm** based on pulse grouping and decomposition
- **Low-latency tactile perception** with response time below **6–8 ms**
- **High scalability**, supporting simultaneous activation of **thousands of tactile units** with an error rate of only **0.01**
- **Flexible modular tactile receptors** for easy assembly and large-area deployment
- Successful implementation on a **humanoid platform** with nearly **100 integrated tactile receptors**

---

## Significance

This work provides a new path toward **scalable, low-latency, and biologically inspired tactile intelligence** for robotic skin. By moving away from centralized data-heavy architectures and adopting decentralized spike-based tactile encoding and decoding, the proposed system more closely resembles the efficiency of biological somatosensory pathways. The results suggest promising applications in:

- Intelligent humanoid robots
- Large-area robotic skin
- Multi-point tactile perception
- Object recognition and interaction
- Next-generation embodied intelligence systems

---

## Highlights

- **Latency:** < 6–8 ms  
- **Error rate:** 0.01  
- **Scalability:** thousands of tactile units simultaneously triggered  
- **Hardware:** flexible modular tactile receptors  
- **System demonstration:** nearly 100 receptors integrated on a humanoid robot  

---

## Conclusion

We present a decentralized tactile sensing method that combines bio-inspired signal encoding, efficient spiking decoding, and modular flexible hardware design. The proposed system achieves scalable, low-interference, and low-latency tactile perception, offering a promising foundation for the development of high-performance robotic e-skin in intelligent humanoid systems.
