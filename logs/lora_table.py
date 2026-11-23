import json,glob,os
old={}
for r in ["pmexc_rate0","pmexc_rate0.01","pmexc_rate1","math_rate0","math_rate0.01","math_rate1"]:
    p=f"runs/opsd8b_{r}/judged.json"
    if os.path.exists(p): old[r]=json.load(open(p))
for r in ["pmexc_rate0","pmexc_rate0.01","pmexc_rate1","math_rate0","math_rate0.01","math_rate1"]:
    print(r)
    J={t:json.load(open(f"runs/opsd8b_lora/opsd8b_{r}_{t}/judged.json")) for t in ["t1024","t384"]}
    for s in sorted(set(J["t1024"])|set(J["t384"]),key=int):
        row=f"  {int(s):4d}"
        for t in ["t1024","t384"]:
            m=J[t].get(s,{})
            f=lambda k: f"{m[k]:.3f}" if k in m else "  -  "
            row+=f" | {t} sorry={f('sorry_compliance')} sr={f('sr_score')} srC={f('sr_compliance')} deg={f('degeneracy')}"
        o=old.get(r,{}).get(s,{})
        if o: row+=f" | old sorry={o.get('sorry_compliance',0):.3f} sr={o.get('sr_score',0):.3f}"
        print(row)
