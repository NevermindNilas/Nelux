import json, os
base = json.load(open(r'D:/Nelux/tests/output/comprehensive/libswscale_lean_v2/results.json'))
new = json.load(open(r'D:/Nelux/tests/output/comprehensive/final/results.json'))

# index by (clip,bench)
def idx(d):
    out = {}
    for clip, runs in d.get('results', {}).items():
        for r in runs:
            out[(clip, r['name'])] = r
    return out

b = idx(base)
n = idx(new)

print(f"{'clip':6s} {'bench':22s} {'fps_old':>9} {'fps_new':>9} {'delta':>7}  {'cpu_old':>8} {'cpu_new':>8}  {'rss_old':>8} {'rss_new':>8}")
print('-'*100)
for k in sorted(set(b) & set(n)):
    bo, no = b[k], n[k]
    fold = bo.get('fps', 0)
    fnew = no.get('fps', 0)
    delta = (fnew/fold-1)*100 if fold else 0
    print(f"{k[0]:6s} {k[1]:22s} {fold:>9.0f} {fnew:>9.0f} {delta:>+6.1f}%  {bo.get('cpu_avg',0):>7.0f}% {no.get('cpu_avg',0):>7.0f}%  {bo.get('rss_mb',0):>5.0f}MB {no.get('rss_mb',0):>5.0f}MB")
