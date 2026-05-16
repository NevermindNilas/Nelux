import json, math
d = json.load(open(r'D:/Nelux/tests/output/pixfmt_matrix/final/results.json'))
print(f"{'label':28s} {'PSNR':>9} {'SSIM':>6} {'VMAF':>7}  {'sync_fps':>8} {'fanout':>7} {'tc':>6} {'ff-null':>7}")
print('-'*90)
for r in d['results']:
    q = r['quality']['nelux']
    p = q['psnr']
    s = q['ssim']
    v = q['vmaf']
    pstr = 'inf' if (isinstance(p, float) and math.isinf(p)) else f"{p:.2f}"
    print(f"{r['label']:28s} {pstr:>9} {s:>6.3f} {v:>7.2f}  {r['nelux-sync_fps']:>8.0f} {r['nelux-fanout_fps']:>7.0f} {r['torchcodec_fps']:>6.0f} {r['ffmpeg-null_fps']:>7.0f}")
