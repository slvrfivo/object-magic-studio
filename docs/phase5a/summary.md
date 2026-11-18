### Environment
- **cpu_fp32_reference**: torch 2.15.0+cpu (CUDA None), CPU  MB, driver -, bf16 hw -, ckpt 7402e0d864fa…
- **gpu_fp32**: torch 2.14.1+cu130 (CUDA 13.0), Tesla T4 7.5 14912MB, driver 580.82.07, bf16 hw False, ckpt 7402e0d864fa…
- **gpu_fp16**: torch 2.14.1+cu130 (CUDA 13.0), Tesla T4 7.5 14912MB, driver 580.82.07, bf16 hw False, ckpt 7402e0d864fa…
- **gpu_fp32_offload**: torch 2.14.1+cu130 (CUDA 13.0), Tesla T4 7.5 14912MB, driver 580.82.07, bf16 hw False, ckpt 7402e0d864fa…

### Model load & single-frame segmentation
| run | load (s) | encoder median (ms) | decoder median (ms) | circle IoU vs GT | IoU vs CPU (min over prompts) |
|---|---|---|---|---|---|
| cpu_fp32_reference | 23.3 | 8,225.6 | 301.8 | 0.9849 | — |
| gpu_fp32 | 10.2 | 160.2 | 9.8 | 0.9849 | 1.0000 |
| gpu_fp16 | 3.6 | 124.7 | 12.2 | 0.9849 | 0.9999 |
| gpu_fp32_offload | 3.2 | — | — | — | — |

### Tracking
| run | frames | avg ms/frame | tracking fps | speed-up vs CPU | init (ms) | total (s) | GPU peak alloc / reserved (MB) | IoU vs GT (min / mean) | IoU vs CPU (min / mean) |
|---|---|---|---|---|---|---|---|---|---|
| cpu_fp32_reference | 60 | 16,820.2 | 0.06 | (CPU) | 11,942.4 | 1,011.5 | — / — | 0.9793 / 0.9891 | — / — |
| gpu_fp32 | 10 | 345.2 | 2.90 | ×53.9 | 99.3 | 3.6 | 734.2 / 984.0 | 0.9792 / 0.9860 | 0.9974 / 0.9990 |
| gpu_fp32 | 30 | 395.1 | 2.53 | ×41.2 | 71.2 | 12.0 | 749.2 / 1,028.0 | 0.9793 / 0.9878 | 1.0000 / 1.0000 |
| gpu_fp32 | 60 | 411.3 | 2.43 | ×40.9 | 77.5 | 24.8 | 771.7 / 1,058.0 | 0.9793 / 0.9891 | 1.0000 / 1.0000 |
| gpu_fp32 | 300 | 438.3 | 2.28 | — | 66.9 | 132.2 | 952.1 / 1,298.0 | 0.9789 / 0.9879 | 1.0000 / 1.0000 |
| gpu_fp16 | 10 | 180.8 | 5.53 | ×102.8 | 85.2 | 2.0 | 585.9 / 720.0 | 0.9790 / 0.9855 | 0.9924 / 0.9976 |
| gpu_fp16 | 30 | 171.9 | 5.82 | ×94.8 | 56.9 | 5.3 | 602.2 / 760.0 | 0.9776 / 0.9849 | 0.9888 / 0.9967 |
| gpu_fp16 | 60 | 180.9 | 5.53 | ×93.0 | 54.1 | 11.1 | 623.7 / 782.0 | 0.9776 / 0.9847 | 0.9888 / 0.9952 |
| gpu_fp16 | 300 | 182.8 | 5.47 | — | 57.6 | 55.7 | 803.8 / 1,012.0 | 0.9776 / 0.9862 | 0.9888 / 0.9952 |
| gpu_fp32_offload | 300 | 438.6 | 2.28 | — | 65.6 | 132.3 | 713.1 / 1,076.0 | 0.9789 / 0.9879 | 1.0000 / 1.0000 |

Speed-up and IoU vs CPU use the first N frames of the 60-frame CPU reference run. Note: SAM 2 uses min(num_frames, 16) object pointers (sam2_base.py), so runs shorter than 16 frames compute slightly differently from longer runs (10-frame run vs first 10 frames of a 60-frame run: IoU ~0.997 even on the same device). Runs of equal length are identical on CPU and GPU fp32 (IoU 1.0).