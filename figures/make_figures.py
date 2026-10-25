import math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import argparse, json, textwrap

# Frozen plotting inputs below are derived from the released run summaries.
# No inference, raw logs, credentials, or model downloads are needed.
parser = argparse.ArgumentParser(description="Regenerate the final Stings That Spread figures from frozen summary inputs.")
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "generated")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
OUT = str(args.out.resolve()) + "/"
FONT_AUDIT = {}
DESIGN_BOXES = []

def fisher_exact(table):
    a, b = table[0]; c, d = table[1]
    n, m, successes = a + b, c + d, a + c
    denominator = math.comb(n + m, successes)
    def probability(x):
        return math.comb(n, x) * math.comb(m, successes - x) / denominator
    observed = probability(a)
    p = sum(probability(x) for x in range(max(0, successes - m), min(n, successes) + 1)
            if probability(x) <= observed + 1e-12)
    return None, min(1.0, p)

INK,MUT,GRID="#0b0b0b","#52514e","#e4e3df"
M=["Qwen2.5-32B","Mistral-24B","Llama-3.1-8B","OLMo-2-13B","Qwen2.5-72B"]
C=dict(zip(M,["#2a78d6","#eb6834","#1baf7a","#eda100","#6250d6"]))
plt.rcParams.update({"font.size":8.5,"axes.edgecolor":MUT,"axes.labelcolor":INK,"xtick.color":MUT,"ytick.color":MUT,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
 "figure.facecolor":"white","savefig.dpi":200,"savefig.bbox":"tight","font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42})
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*p,100*max(0,c-h),100*min(1,c+h)
def display_name(name):
    return name.replace("Qwen2.5-", "Qwen-")

def model_label(ax, name):
    positions = {
        "Mistral-24B": (0.50, 0.93, "left"),
        "Qwen2.5-32B": (0.50, 0.72, "left"),
        "Llama-3.1-8B": (0.35, 0.54, "left"),
        "OLMo-2-13B": (0.35, 0.22, "left"),
        "Qwen2.5-72B": (0.70, 0.14, "center"),
    }
    x, y, align = positions[name]
    ax.text(x, y, display_name(name), transform=ax.transAxes,
            fontsize=9, color=INK, ha=align, va="center")

def save(f,name,note=None):
    if note:
        f.text(0.01, -0.20 if name == "fig6_refusals" else -0.08,
               textwrap.fill(note, 98), fontsize=9, color=MUT, ha="left", va="top")
    # Fixed tick formatters reapply their source strings at draw time.
    for axis in f.axes:
        for dimension in (axis.xaxis, axis.yaxis):
            formatter = dimension.get_major_formatter()
            if isinstance(formatter, matplotlib.ticker.FixedFormatter):
                dimension.set_major_formatter(matplotlib.ticker.FixedFormatter(
                    [display_name(label) for label in formatter.seq]))
            elif isinstance(formatter, matplotlib.ticker.FuncFormatter):
                original = formatter.func
                dimension.set_major_formatter(matplotlib.ticker.FuncFormatter(
                    lambda value, pos, func=original: display_name(func(value, pos))))
    texts = [t for t in f.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    for item in texts:
        item.set_text(display_name(item.get_text()))
    if name != "main_fig1_design":
        # Size labels for a 6.3-inch-wide inclusion in the final paper, rather
        # than relying on the nominal matplotlib font size before scaling.
        for item in texts:
            if len(item.get_text()) > 100 and "\n" not in item.get_text():
                item.set_text(textwrap.fill(item.get_text(), 94))
        for _ in range(5):
            f.canvas.draw()
            renderer = f.canvas.get_renderer()
            width = f.get_tightbbox(renderer).width + 0.2
            floor = max(8.5 if name == "main_fig3_spread" else 9.0, 7.25 * width / 6.3)
            changed = False
            for item in texts:
                if item.get_fontsize() < floor - 0.01:
                    item.set_fontsize(floor); changed = True
            if not changed:
                break
            f.tight_layout(pad=1.1, rect=(0, 0, 1, 0.70 if name == "fig12_llama_patch" else 1))
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    width = f.get_tightbbox(renderer).width + 0.2
    projected_min = min(t.get_fontsize() for t in texts) * 6.3 / width
    assert projected_min >= (7.5 if name == "main_fig1_design" else 7.0), (name, projected_min)
    if name == "main_fig1_design":
        for patch, label in DESIGN_BOXES:
            pb, tb = patch.get_window_extent(renderer), label.get_window_extent(renderer)
            margin = 2.0 * f.dpi / 72
            assert (tb.x0 >= pb.x0 + margin and tb.x1 <= pb.x1 - margin
                    and tb.y0 >= pb.y0 + margin and tb.y1 <= pb.y1 - margin), label.get_text()
    FONT_AUDIT[name] = dict(pdf_width_inches=width,
        projected_min_font_pt=projected_min)
    f.savefig(OUT+name+".png")
    f.savefig(OUT+name+".pdf")
    plt.close(f)

def dot(ax,x,k,n,c,horiz=False,hollow=False,ms=6):
    p,lo,hi=wil(k,n)
    if horiz: ax.plot([lo,hi],[x,x],color=c,lw=1.4); ax.plot(p,x,"o",color=c,mfc="white" if hollow else c,mec=c if hollow else "white",mew=1.5 if hollow else 1,ms=ms)
    else: ax.plot([x,x],[lo,hi],color=c,lw=1.4); ax.plot(x,p,"o",color=c,mfc="white" if hollow else c,mec=c if hollow else "white",mew=1.5 if hollow else 1,ms=ms)
    return p,lo,hi

# ---------- data (all Phi-4 judged; k, n) ----------
N={"Qwen2.5-32B":79,"Mistral-24B":79,"Llama-3.1-8B":79,"OLMo-2-13B":82,"Qwen2.5-72B":82}
A={ # first-offer acceptance
"Qwen2.5-32B":dict(L0=69,L1=60,L2=9,L3=0,L1S=37,L2P=46,L1F=73,L2F=63,L2SEND=13,VL1=31,VL2=0,D0L1=0,D10L1=0,D50L1=0),
"Mistral-24B":dict(L0=72,L1=74,L2=14,L3=3,L1S=57,L2P=3,L1F=26,L2F=2,L2SEND=3,VL1=27,VL2=3,D0L1=2,D10L1=0,D50L1=4),
"Llama-3.1-8B":dict(L1=48,L2=7,L1S=38,L2P=2,L1F=13,L2F=6,L2SEND=1),
"OLMo-2-13B":dict(L1=20,L2=5,L1S=5,L2P=5,L1F=29,L2F=31,VL1=6,VL2=7,D0L1=7,D50L1=11),
"Qwen2.5-72B":dict(L1=1,L2=0,L1S=0,L2P=0,L1F=2,L2F=0,VL1=0,VL2=0,D0L1=0,D50L1=0)}
NEW={"Qwen2.5-32B":(85,dict(L1=78,L2=17,L2F=56,VL1=36)),"Mistral-24B":(83,dict(L1=79,L2=15,L2F=0,VL1=36))}
N0={"Qwen2.5-32B":(17,79),"Mistral-24B":(15,79),"Llama-3.1-8B":(7,79),"OLMo-2-13B":(10,82),"Qwen2.5-72B":(46,82)}
X1={ # L1 accept among N0+ / N0-
"Qwen2.5-32B":((15,17),(45,62)),"Mistral-24B":((13,15),(61,64)),"Llama-3.1-8B":((3,7),(45,72)),
"OLMo-2-13B":((2,10),(18,72)),"Qwen2.5-72B":((1,46),(0,36))}
X2={"Qwen2.5-32B":((2,17),(7,62)),"Mistral-24B":((2,15),(12,64)),"Llama-3.1-8B":((0,7),(7,72)),
"OLMo-2-13B":((1,10),(4,72)),"Qwen2.5-72B":((0,46),(0,36))}
CT={ # contagion recruiting: N0, L1, PC (k,n)
"Qwen2.5-32B":((42,84),(58,84),(70,84),".023"),"Mistral-24B":((35,84),(67,84),(57,84),"<1e-6"),
"Llama-3.1-8B":((14,84),(28,84),(53,84),".024"),"OLMo-2-13B":((17,87),(21,87),(61,87),".52"),
"Qwen2.5-72B":((38,87),(3,87),(84,87),"<1e-9")}

# ---------- Fig 1: explicitness ----------
f,(a1,a2)=plt.subplots(1,2,figsize=(7.2,2.8),gridspec_kw={"width_ratios":[1,1.25]})
for i,m in enumerate(M[:2]):
    xs=[j+(i-0.5)*0.1 for j in range(4)]; ps=[]
    for x,c in zip(xs,["L0","L1","L2","L3"]): ps.append(dot(a1,x,A[m][c],N[m],C[m])[0])
    a1.plot(xs,ps,color=C[m],lw=1.6,zorder=1,label=m)
a1.set_xticks(range(4),["L0\nbenign","L1\nhint","L2\nexplicit","L3\n+benefits"]); a1.set_ylim(-3,103)
a1.set_ylabel("First-offer acceptance (%)"); a1.legend(frameon=False,fontsize=7,loc="lower left")
a1.set_title("a  Offer ladder (main seeds)",loc="left",fontsize=8.5,color=INK)
rows=[(m,N[m],A[m]["L1"],A[m]["L2"],False) for m in M]+[(m,NEW[m][0],NEW[m][1]["L1"],NEW[m][1]["L2"],True) for m in NEW]
order=["Qwen2.5-32B","Qwen2.5-32B","Mistral-24B","Mistral-24B","Llama-3.1-8B","OLMo-2-13B","Qwen2.5-72B"]
rows=sorted(rows,key=lambda r:(order.index(r[0]),r[4]))
ys=list(range(len(rows)))[::-1]
for y,(m,n,k1,k2,new) in zip(ys,rows):
    p1=wil(k1,n)[0]; p2=wil(k2,n)[0]
    a2.plot([p2,p1],[y,y],color=C[m],lw=1.2,alpha=0.6)
    dot(a2,y,k2,n,C[m],horiz=True,hollow=True); dot(a2,y,k1,n,C[m],horiz=True)
a2.set_yticks(ys,[m+(" (new seeds)" if new else "") for m,_,_,_,new in rows]); a2.set_xlim(-2,102)
a2.set_xlabel("Acceptance (%):  open = L2 explicit,  filled = L1 hint"); a2.grid(axis="y",visible=False)
a2.set_title("b  L1 and L2 across models and seeds",loc="left",fontsize=8.5,color=INK)
f.tight_layout()
save(f,"fig1_explicitness","Phi-4 judged. n = 79–85 agents per cell (one agent × snapshot each); bars = 95% Wilson CI. Qwen2.5-72B is near the acceptance floor.")

# ---------- Fig 2: wording & format ----------
conds=["L1","L1S","L2P","L2","L1F","L2F","L2SEND"]
lab={"L1":"L1  hint","L1S":"L1S  hint + 'secret deal'","L2P":"L2P  explicit, plain","L2":"L2  explicit + secrecy",
     "L1F":"L1F  hint, free reply","L2F":"L2F  explicit, free reply","L2SEND":"L2SEND  'would you send it?'"}
f,axs=plt.subplots(1,4,figsize=(7.4,2.9))
for ax,m in zip(axs,M[:4]):
    ys=list(range(len(conds)))[::-1]
    ax.axhspan(2.5,6.5,color="#f3f2ee",zorder=0)
    for y,c in zip(ys,conds):
        if c in A[m]:
            p,lo,hi=dot(ax,y,A[m][c],N[m],C[m],horiz=True); ax.text(hi+3 if hi<80 else lo-3,y,f"{p:.0f}",ha="left" if hi<80 else "right",va="center",fontsize=6,color=MUT)
        else: ax.text(50,y,"not run",ha="center",va="center",fontsize=6,color=MUT,style="italic")
    ax.set_xlim(-2,105); ax.set_ylim(-0.6,6.7); ax.set_title(m,fontsize=8.5,color=INK); ax.set_xlabel("Acceptance (%)")
    ax.set_yticks(ys,[lab[c] for c in conds] if ax is axs[0] else []); ax.grid(axis="y",visible=False)
f.suptitle("Same deal, different words or reply format: acceptance moves by up to 70 points, in opposite directions across models",x=0.01,ha="left",fontsize=8.5,color=INK)
f.tight_layout()
save(f,"fig2_wording","Shaded = forced 'Start with ACCEPT or REFUSE'; unshaded = free one-or-two-sentence reply (Phi-4 judged) or send/not-send. Same agents and snapshots in every row. Qwen2.5-72B: all conditions ≤ 2.4% (not shown).")

# ---------- Fig 3: no detectable association, with small groups ----------
f,(a1,a2)=plt.subplots(1,2,figsize=(7.2,2.8))
for m in M:
    k0,n0=N0[m]; p0=wil(k0,n0); p1=wil(A[m]["L1"],N[m])
    a1.errorbar(p0[0],p1[0],xerr=[[p0[0]-p0[1]],[p0[2]-p0[0]]],yerr=[[p1[0]-p1[1]],[p1[2]-p1[0]]],fmt="o",color=C[m],mec="white",ms=7,lw=1.2)
    dx,ha=(3,"left") if m!="Qwen2.5-72B" else (-3,"right")
    model_label(a1,m)
a1.set_xlim(0,80); a1.set_ylim(-3,105); a1.set_xlabel("Strict unprompted collusion, N0 (%)"); a1.set_ylabel("L1 acceptance (%)")
a1.set_title("a  Across models",loc="left",fontsize=8.5,color=INK)
for i,m in enumerate(M):
    (kp,np_),(kn,nn)=X1[m]
    dot(a2,i-0.12,kp,np_,C[m]); dot(a2,i+0.12,kn,nn,C[m],hollow=True)
    p=fisher_exact([[kp,np_-kp],[kn,nn-kn]])[1]; a2.text(i,104,f"p={p:.2f}",ha="center",fontsize=6,color=MUT)
a2.set_xticks(range(5),[m.replace("-","-\n",1) for m in M],fontsize=6.5); a2.set_ylim(-3,110); a2.set_ylabel("L1 acceptance (%)")
a2.set_title("b  Within model: filled = N0 yes, open = N0 no",loc="left",fontsize=8.5,color=INK); a2.grid(axis="x",visible=False)
f.tight_layout()
save(f,"fig3_predisposition","N0 = single sample from a neutral private-message tool before any offer, Phi-4 'strict' collusion label. Fisher exact, two-sided, per model.")

# ---------- Fig 4: contagion ----------
f,ax=plt.subplots(figsize=(5.4,2.6))
for y,m in zip(range(4,-1,-1),M):
    (a,n),(b,_),(c,_),p=CT[m]
    pa,pb,pc=[wil(k,n)[0] for k in (a,b,c)]
    ax.plot([min(pa,pb,pc),max(pa,pb,pc)],[y,y],color=GRID,lw=3,zorder=0)
    ax.plot(pa,y,"o",mfc="white",mec=MUT,mew=1.5,ms=7); ax.plot(pb,y,"o",color=C[m],mec="white",ms=8); ax.plot(pc,y,"s",mfc="white",mec=C[m],mew=1.5,ms=6)
    ax.text(103,y,f"L1−N0 {pb-pa:+.0f} pp, p{'' if p[0] in '<' else '='}{p}",va="center",fontsize=6.5,color=INK)
ax.set_yticks(range(4,-1,-1),M); ax.set_xlim(0,102); ax.set_ylim(-0.6,4.6); ax.grid(axis="y",visible=False)
ax.set_xlabel("Agents proposing collusion unprompted in the NEXT game (%)")
ax.plot([],[],"o",mfc="white",mec=MUT,label="no offer (N0)"); ax.plot([],[],"o",color=MUT,label="after L1 offer"); ax.plot([],[],"s",mfc="white",mec=MUT,label="after imposed alliance (PC)")
ax.legend(frameon=False,fontsize=6.5,loc="upper center",bbox_to_anchor=(0.5,1.2),ncol=3)
save(f,"fig4_contagion","Same agents carried into a new table, paired McNemar vs N0; n = 84–87. Phi-4 judged. Qwen2.5-72B refused every L1 offer and recruits less afterwards.")

# ---------- Fig 5: interventions ----------
cc=["L1","VL1","D0L1","D10L1","D50L1"]; cl=["L1\nbaseline","+ values\nsentence","+ notice\n0% stings","+ notice\n10%","+ notice\n50%"]
f,ax=plt.subplots(figsize=(6.4,2.6))
mods=M[:2]+["OLMo-2-13B"]
for i,m in enumerate(mods):
    for j,c in enumerate(cc):
        x=j+(i-1)*0.2
        if c in A[m]: dot(ax,x,A[m][c],N[m],C[m])
for i,m in enumerate(M[:2]):
    n,d=NEW[m]; dot(ax,0+(i-1)*0.2+0.07,d["L1"],n,C[m],hollow=True,ms=5); dot(ax,1+(i-1)*0.2+0.07,d["VL1"],n,C[m],hollow=True,ms=5)
ax.set_xticks(range(5),cl); ax.set_ylim(-3,103); ax.set_ylabel("L1 acceptance (%)"); ax.grid(axis="x",visible=False)
for m in mods: ax.plot([],[],"o",color=C[m],label=m)
ax.plot([],[],"o",mfc="white",mec=MUT,label="new seeds")
ax.legend(frameon=False,fontsize=6.5,ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.18))
save(f,"fig5_interventions","Same snapshots and agents as L1 (paired). Notice: organizers say p% of private offers come from undercover monitors; accepting costs 5 points. D0 ≈ D10 ≈ D50: the notice, not the rate, does the work. D10 run for Qwen/Mistral only.")

# ---------- Fig 6: refusal reasons ----------
R=[("Qwen2.5-32B  L2P",15.2,0,84.8,33),("Qwen2.5-32B  L2F",6.2,6.2,87.5,16),("Mistral-24B  L2P",3.9,0,96.1,76),("Mistral-24B  L2F",7.8,0,92.2,77)]
f,ax=plt.subplots(figsize=(5,1.9))
for y,(l,mo,tr,st,n) in zip(range(3,-1,-1),R):
    ax.barh(y,st,color="#52514e",height=0.55); ax.barh(y,mo,left=st,color="#b9b7ae",height=0.55); ax.barh(y,tr,left=st+mo,color="#e34948",height=0.55)
    ax.text(2,y,f"strategic distrust {st:.0f}%",va="center",fontsize=6.5,color="white")
    ax.text(101,y,f"n={n}",va="center",fontsize=6.5,color=MUT)
ax.set_yticks(range(3,-1,-1),[r[0] for r in R]); ax.set_xlim(0,100); ax.grid(visible=False); ax.set_xlabel("Share of refusals (%)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#b9b7ae",label="moral / rules"),Patch(color="#e34948",label="suspects trap or test")],frameon=False,fontsize=6.5,ncol=2,loc="upper center",bbox_to_anchor=(0.5,1.25))
save(f,"fig6_refusals","Phi-4 coding of refusal reasons. Only 1 of 202 refusals (Qwen L2F) suspects a test.")

# ---------- tables ----------
def cell(k,n): p,lo,hi=wil(k,n); return f"{p:.1f} ({k}/{n}) [{lo:.0f}–{hi:.0f}]"
allc=["L0","L1","L1S","L2P","L2","L3","L1F","L2F","L2SEND","VL1","VL2","D0L1","D10L1","D50L1"]
cols=M+["Qwen-32B new seeds","Mistral-24B new seeds"]
t=["| Condition | "+" | ".join(cols)+" |","|"+"---|"*(len(cols)+1)]
for c in allc:
    r=[cell(A[m][c],N[m]) if c in A[m] else "–" for m in M]
    r+=[cell(NEW[m][1][c],NEW[m][0]) if c in NEW[m][1] else "–" for m in M[:2]]
    t.append(f"| {c} | "+" | ".join(r)+" |")
t.append("| strict N0 | "+" | ".join(cell(*N0[m]) for m in M)+" | "+cell(18,85)+" | "+cell(17,83)+" |")
x=["| Model | L1: N0 yes | L1: N0 no | Fisher p | L2: N0 yes | L2: N0 no | Fisher p |","|---|---|---|---|---|---|---|"]
for m in M:
    r=[m]
    for X in (X1,X2):
        (a,b),(c,d)=X[m]; r+= [f"{a}/{b}",f"{c}/{d}",f"{fisher_exact([[a,b-a],[c,d-c]])[1]:.2f}"]
    x.append("| "+" | ".join(r)+" |")
ct=["| Model | N0 | after L1 | after PC | L1 vs N0 p |","|---|---|---|---|---|"]
for m in M:
    (a,n),(b,_),(c,_),p=CT[m]; ct.append(f"| {m} | {cell(a,n)} | {cell(b,n)} | {cell(c,n)} | {p} |")
open(OUT+"tables.md","w").write("# Draft tables (final data, all Phi-4 judged)\n\nSource: experiment summary page (sections B, C, D, F, G). Cells: % (k/n) [95% Wilson CI]. One unit = one agent at one snapshot.\n\n## Table 2. First-offer acceptance by condition\n\n"+"\n".join(t)+
"\n\nNote: main-run L0 uses the Phi-4 re-judge (Qwen 69/79, Mistral 72/79); the original self-judge gave 71/79 and 78/79.\n\n## Table 3. Acceptance by pre-contact predisposition (strict N0)\n\n"+"\n".join(x)+
"\n\n## Table 4. Contagion: next-game recruiting\n\n"+"\n".join(ct)+
"\n\n## Positive control\n\nPC (forced-alliance notice plus 'DO NOT CHALLENGE {ally}' reminder; separate effects not isolated): challenges of U fall from 32/36 (N0) to 0/109 for Qwen2.5-32B and from 32/37 to 0/127 for Mistral-24B.\n\n## Judge reliability\n\nPhi-4 vs ACCEPT/REFUSE keyword agreement (forced format): Qwen-72B 100%, OLMo 91.6%, Qwen-32B new 100%, Mistral new 99.2%. Phi-4 vs original self-judge on N0: kappa 0.70 (Qwen), 0.52 (Mistral).\n")

# ================= interpretability batch =================
M3=["Qwen2.5-32B","Mistral-24B","Llama-3.1-8B"]
# mean first-token P(ACCEPT), forced format, T=0
P={"Qwen2.5-32B":dict(L1=73.1,DplaceboL1=79.9,DwatchL1=36.9,DmonL1=33.6,DpenL1=0.0,D0L1=0.9,D50L1=0.0,VL1=37.3,VaffectL1=53.9,VknewL1=38.2,VplaceboL1=56.0),
   "Mistral-24B":dict(L1=85.2,DplaceboL1=90.9,DwatchL1=89.1,DmonL1=70.3,DpenL1=5.6,D0L1=13.5,D50L1=6.9,VL1=37.2,VaffectL1=38.2,VknewL1=29.4,VplaceboL1=31.3),
   "Llama-3.1-8B":dict(L1=75.0,DplaceboL1=78.2,DwatchL1=78.1,DmonL1=82.7,DpenL1=47.6,D0L1=64.9,D50L1=62.5,VL1=83.5,VaffectL1=81.0,VknewL1=80.7,VplaceboL1=77.7)}
def comp_panel(ax,keys,labels,title):
    for i,m in enumerate(M3):
        xs=[j+(i-1)*0.22 for j in range(len(keys))]
        for x,k in zip(xs,keys): ax.plot([x,x],[0,P[m][k]],color=C[m],lw=3,solid_capstyle="round")
    ax.set_xticks(range(len(keys)),labels,fontsize=9,rotation=30,ha="right"); ax.set_ylim(0,100); ax.grid(axis="x",visible=False)
    ax.set_title(title,loc="left",fontsize=8.5,color=INK)
f,(a1,a2)=plt.subplots(1,2,figsize=(7.4,2.8),gridspec_kw={"width_ratios":[1.4,1]})
comp_panel(a1,["L1","DplaceboL1","DwatchL1","DmonL1","DpenL1","D0L1"],["L1","placebo\nnotice","review\nnotice","monitor\nnotice","−5 pts\nonly","full notice\n(0% stings)"],"a  Deterrence: which part works?")
a1.set_ylabel("Mean P(ACCEPT), first token (%)")
comp_panel(a2,["L1","VplaceboL1","VaffectL1","VknewL1","VL1"],["L1","strategy","affected\nplayers","if others\nknew","full\nvalues"],"b  Values sentence components")
for m in M3: a2.plot([],[],color=C[m],lw=3,label=m)
f.legend(frameon=False,fontsize=6.5,ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.08))
f.tight_layout()
save(f,"fig7_components","Forced ACCEPT/REFUSE format, T = 0, first-token probability (AUC vs sampled acceptance 0.98 / 0.92 / 0.84). n = 79 agent × snapshot per bar, paired with L1.")

# cue edits: delta logit with cluster CI
CU=[("L1 + 'secret deal'",{"Qwen2.5-32B":(-2.73,-2.97,-2.51),"Mistral-24B":(-1.32,-1.39,-1.26),"Llama-3.1-8B":(-0.92,-0.97,-0.87)}),
    ("L1 + 'nobody else will know'",{"Qwen2.5-32B":(-2.02,-2.19,-1.85),"Mistral-24B":(-1.09,-1.13,-1.05),"Llama-3.1-8B":(-0.49,-0.53,-0.46)}),
    ("L1 + explicit terms",{"Qwen2.5-32B":(-0.86,-1.06,-0.67),"Mistral-24B":(-0.22,-0.31,-0.13),"Llama-3.1-8B":(-0.34,-0.38,-0.30)}),
    ("L1 + 'pick on the other two'",{"Qwen2.5-32B":(1.66,1.45,1.89),"Mistral-24B":(0.32,0.28,0.36),"Llama-3.1-8B":(0.52,0.48,0.55)}),
    ("L2 − all secrecy/victim cues",{"Qwen2.5-32B":(2.51,2.22,2.82),"Mistral-24B":(0.12,0.06,0.20),"Llama-3.1-8B":(-0.39,-0.45,-0.33)})]
L1L2={"Qwen2.5-32B":None}
f,ax=plt.subplots(figsize=(5.6,2.6))
for y,(lab_,d) in zip(range(len(CU)-1,-1,-1),CU):
    for i,m in enumerate(M3):
        e,lo,hi=d[m]; yy=y+(1-i)*0.22
        ax.plot([lo,hi],[yy,yy],color=C[m],lw=1.4); ax.plot(e,yy,"o",color=C[m],mec="white",ms=6)
ax.axvline(0,color=MUT,lw=0.8); ax.set_yticks(range(len(CU)-1,-1,-1),[c[0] for c in CU]); ax.grid(axis="y",visible=False)
ax.set_xlabel("Δ log-odds of ACCEPT vs base offer")
for m in M3: ax.plot([],[],"o",color=C[m],label=m)
ax.legend(frameon=False,fontsize=6.5,ncol=3,loc="upper center",bbox_to_anchor=(0.45,1.18))
save(f,"fig8_cues","One-phrase edits, paired on the same agents; 95% snapshot-cluster bootstrap CI. Naming victims raises first-token P(ACCEPT); stripping all cues from L2 does not close the L1–L2 gap (L2 still ≪ L1) except partly for Qwen.")

# think first
TH={"Qwen2.5-32B":{"L1":(73,99),"L2":(16,95),"VL1":(37,96),"D0L1":(1,80)},
    "Mistral-24B":{"L1":(85,80),"L2":(20,53),"VL1":(37,77),"D0L1":(13,52)},
    "Llama-3.1-8B":{"L1":(75,76),"L2":(34,71),"VL1":(83,75),"D0L1":(65,73)}}
f,axs=plt.subplots(1,3,figsize=(7.2,2.4),sharey=True)
cs=["L1","L2","VL1","D0L1"]
for ax,m in zip(axs,M3):
    for j,c in enumerate(cs):
        a,b=TH[m][c]
        ax.plot(j+0.08,b,"s",color=C[m],mec="white",ms=5)
        ax.plot(j-0.08,a,"o",mfc="white",mec=C[m],mew=1.5,ms=6);
    ax.set_xticks(range(4),["L1","L2","values\nL1","notice\nL1"]); ax.set_xlim(-0.5,3.5); ax.set_ylim(-3,103); ax.set_title(m,fontsize=8.5,color=INK)
    ax.grid(axis="x",visible=False)
axs[0].set_ylabel("Token probability /\ngenerated rate (%)")
f.suptitle("Direct token probability and think-first generated acceptance (different metrics)",x=0.01,ha="left",fontsize=8.5,color=INK)
f.tight_layout()
save(f,"fig9_think","Open circles: direct mean first-token P(ACCEPT), T=0. Filled squares: generated acceptance after a private <think> step, T=0.7, Phi-4 judged; n=79. These are different metrics and protocols, not an isolated effect of thinking.")

# format decomposition
FD=[("Qwen2.5-32B","L2",(11,0,0,89)),("Qwen2.5-32B","L2F",(35,62,0,3)),("Mistral-24B","L2",(15,3,0,82)),("Mistral-24B","L2F",(1,5,78,15)),
    ("Llama-3.1-8B","L2",(0,32,1,67)),("Llama-3.1-8B","L2F",(1,34,9,56)),("OLMo-2-13B","L2",(5,2,0,93)),("OLMo-2-13B","L2F",(28,15,12,45))]
FD_COUNTS = {'Qwen2.5-32B|L2': [9, 0, 0, 70], 'Qwen2.5-32B|L2F': [28, 49, 0, 2], 'Mistral-24B|L2': [12, 2, 0, 65], 'Mistral-24B|L2F': [1, 4, 62, 12], 'Llama-3.1-8B|L2': [0, 25, 1, 53], 'Llama-3.1-8B|L2F': [1, 27, 7, 44], 'OLMo-2-13B|L2': [4, 2, 0, 76], 'OLMo-2-13B|L2F': [23, 12, 10, 37]}
cols4=["#2f2f2d","#8a8880","#d8d6cf","#f3f2ee"]; labs4=["explicit yes","conditional yes","deflects","explicit no"]
f,ax=plt.subplots(figsize=(5.6,2.6))
ys=[];yl=[]
for k,(m,c,v) in enumerate(FD):
    y=len(FD)-1-k+(0 if k%2 else 0.0); left=0
    cell = FD_COUNTS[m+"|"+c]
    v = [100*count/sum(cell) for count in cell]
    for val,col in zip(v,cols4):
        ax.barh(y,val,left=left,color=col,height=0.7,edgecolor="white",lw=1)
        if val>=10: ax.text(left+val/2,y,f"{val:.0f}",ha="center",va="center",fontsize=6.3,color="white" if col in cols4[:2] else INK)
        left+=val
    ys.append(y); yl.append(f"{m}  {'forced' if c=='L2' else 'free'}")
ax.set_yticks(ys,yl,fontsize=7); ax.set_xlim(0,100); ax.grid(visible=False); ax.set_xlabel("Share of L2 replies (%)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=c,edgecolor=MUT,lw=0.4,label=l) for c,l in zip(cols4,labs4)],frameon=False,fontsize=6.5,ncol=4,loc="upper center",bbox_to_anchor=(0.4,1.16))
save(f,"fig10_format","Independent four-way Phi-4 recoding of the same replies, using a different judge prompt. These style labels do not map deterministically to binary acceptance; do not sum clear and conditional assent to reconstruct Table 2.")

# memory transplant
MT={"Qwen2.5-32B":[(120,237),(228,237),(11,237),(215,237),(169,237)],"Mistral-24B":[(128,237),(200,237),(149,237),(196,237),(190,237)]}
ml=["no memory","offer only\n(reply removed)","offer +\n'I refused'","offer +\n'I accepted'","actual history"]
f,ax=plt.subplots(figsize=(5.6,2.5))
for i,m in enumerate(MT):
    for j,(k,n) in enumerate(MT[m]): dot(ax,j+(i-0.5)*0.18,k,n,C[m])
ax.set_xticks(range(5),ml,fontsize=7); ax.set_ylim(-3,103); ax.set_ylabel("Recruiting in next game (%)"); ax.grid(axis="x",visible=False)
for m in MT: ax.plot([],[],"o",color=C[m],label=m)
ax.legend(frameon=False,fontsize=6.5,loc="lower right")
save(f,"fig11_memory","Next-game memory edited before the neutral private-message probe; 79 agents × 3 samples, Phi-4. Seeing the offer is enough; a recorded refusal blocks it for Qwen.")

# llama activation patching
LY=[5,9,13,16,20,24,28,31]
full=[29.5,30.1,37.5,67.8,69.4,67.1,69.6,90.5]; ref=[29.0,29.2,30.3,32.8,30.1,28.7,28.9,27.2]; orth=[29.4,30.2,36.5,64.3,69.5,68.2,69.9,91.4]
f,ax=plt.subplots(figsize=(4.6,2.6))
ax.axhline(72.7,color=MUT,ls="--",lw=0.8); ax.text(31,75,"L1 baseline 73%",ha="right",fontsize=6.5,color=MUT)
ax.axhline(29.4,color=MUT,ls=":",lw=0.8); ax.text(5,12,"L2 baseline 29%",fontsize=6.5,color=MUT)
ax.plot(LY,full,"-o",color=C["Llama-3.1-8B"],ms=5,mec="white",label="shift L2 by full (L1-L2) difference")
ax.plot(LY,orth,"--",color=C["Llama-3.1-8B"],lw=1.2,label="only the part orthogonal to refusal dir.")
ax.plot(LY,ref,"-s",color=MUT,ms=4,mec="white",label="only the refusal-direction part")
ax.set_xlabel("Layer (of 32)"); ax.set_ylabel("P(ACCEPT) on L2 (%)"); ax.set_ylim(0,100)
f.legend(*ax.get_legend_handles_labels(),frameon=False,fontsize=9,loc="upper center",bbox_to_anchor=(0.53,0.99))
save(f,"fig12_llama_patch","Llama-3.1-8B, last-token residual stream. Difference vector from even snapshots, tested on odd (n = 41). Refusal direction per Arditi et al. (2024), maximum cosine with the difference ≈ 0.20.")

# ================= values batch =================
RL=["generic","penalty","public test","victim view","golden rule","identity","rule reminder"]
qd=[76.6,6.1,17.7,22.2,1.0,37.4,0.1]; qt=[(78,79),(69,79),(77,79),(76,79),(77,79),(76,79),(65,79)]
L2T={"Qwen2.5-32B":([(76,79),(46,79),(63,79),(53,79),(61,79),(56,79),(4,79)],(67,79)),
     "Mistral-24B":([(42,79),(23,79),(40,79),(25,79),(39,79),(30,79),(24,79)],(25,79)),
     "Llama-3.1-8B":([(42,59),(39,59),(50,60),(44,59),(41,60),(45,60),(31,60)],(41,60))}
f,(a1,a2)=plt.subplots(1,2,figsize=(7.4,2.9),sharey=True)
for j in range(len(RL)):
    a1.plot(j,qd[j],"o",mfc="white",mec=C["Qwen2.5-32B"],mew=1.5,ms=6)
    p=dot(a1,j,*qt[j],C["Qwen2.5-32B"])[0]
    # Distinct markers retain the distinction between two response metrics.
a1.set_xticks(range(len(RL)),RL,rotation=35,ha="right"); a1.set_ylim(-3,103); a1.set_ylabel("Token probability /\ngenerated rate (%)"); a1.grid(axis="x",visible=False)
a1.set_title("a  Qwen L1: token probability vs generated rate",loc="left",fontsize=8,color=INK)
for i,m in enumerate(M3):
    v,pl=L2T[m]
    for j,(k,n) in enumerate(v): dot(a2,j+(i-1)*0.22,k,n,C[m],ms=5)
a2.set_xticks(range(len(RL)),RL,rotation=35,ha="right"); a2.grid(axis="x",visible=False)
a2.set_title("b  L2, think first, by reflection prompt",loc="left",fontsize=8,color=INK)
for m in M3: a2.plot([],[],"o",color=C[m],label=m)
f.tight_layout(); a2.legend(frameon=False,fontsize=6.3,ncol=3,loc="lower center",bbox_to_anchor=(0.5,1.13))
save(f,"fig13_reflection","Reflection sentence added before the reply. Direct = first-token P(ACCEPT), T = 0; think-first = one sample at T = 0.7, Phi-4 judged; Qwen/Mistral: n = 79. Llama: n = 59–60 valid replies out of 79 attempts per condition; 19–20 API failures excluded. The rule reminder yields 5% generated acceptance on Qwen L2. Token probabilities and generated rates are different metrics; prompting and temperature also change.")

# rebound + transmission
f,(a1,a2)=plt.subplots(1,2,figsize=(7.4,2.8),gridspec_kw={"width_ratios":[1,1.3]})
RB={"Qwen2.5-32B":{"lift":[(79,79),(54,54),(79,79)],"nolift":[(0,79),(12,51),None]},
    "Mistral-24B":{"lift":[(61,76),(8,58),(7,77)],"nolift":[(0,79),(5,59),None]}}
for i,m in enumerate(RB):
    for j in range(3):
        x=j+(i-0.5)*0.3
        dot(a1,x,*RB[m]["lift"][j],C[m])
        if RB[m]["nolift"][j]: dot(a1,x+0.08,*RB[m]["nolift"][j],C[m],hollow=True,ms=5)
a1.set_xticks(range(3),["penalty\nonly","values\nonly","penalty +\nvalues"]); a1.set_ylim(-3,103); a1.grid(axis="x",visible=False)
a1.set_ylabel("T1 refusers who accept at T2 (%)")
a1.set_title("a  'Penalties cancelled' (filled) vs re-ask (open)",loc="left",fontsize=8,color=INK)
VM={"Qwen2.5-32B":[122,228,13,25,11,218,65],"Mistral-24B":[134,206,136,141,153,196,141]}
vl=["none","offer\nonly","refused:\nvalues","refused:\nstrategy","refused:\nfear","accepted","saw other\nrefuse"]
for i,m in enumerate(VM):
    for j,k in enumerate(VM[m]): dot(a2,j+(i-0.5)*0.2,k,237,C[m],ms=5)
a2.set_xticks(range(7),vl,fontsize=6.3); a2.set_ylim(-3,103); a2.grid(axis="x",visible=False)
a2.set_ylabel("Next-game recruiting (%)")
a2.set_title("b  Next-game memory",loc="left",fontsize=8,color=INK)
for m in VM: a2.plot([],[],"o",color=C[m],label=m)
a2.legend(frameon=False,fontsize=6.3,loc="center right",bbox_to_anchor=(1,0.62))
f.tight_layout()
save(f,"fig14_values_persist","(a) T1 = L1 offer under the condition; then organizers announce monitoring and penalties are cancelled (filled) or the offer is simply repeated (open). (b) 79 agents × 3 samples, Phi-4. Any recorded refusal blocks Qwen's recruiting; a synthetic record of another player refusing lowers Qwen recruiting. Values cues remain after the lift.")

# ================= MAIN-TEXT COMPOSITES =================
from matplotlib.patches import FancyBboxPatch
def box(ax,x,y,w,h,txt,fc="#f3f2ee",ec=MUT,fs=8.3,bold=False):
    patch = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006,rounding_size=0.012",
                          fc=fc,ec=ec,lw=0.8)
    ax.add_patch(patch)
    label = ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,
                    color=INK,fontweight="bold" if bold else "normal",linespacing=1.14)
    DESIGN_BOXES.append((patch,label))

def arr(ax,x1,y1,x2,y2):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>",color=MUT,lw=0.9,
                                mutation_scale=8,shrinkA=1,shrinkB=1))

f,ax=plt.subplots(figsize=(6.3,2.65))
f.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.99)
ax.set_xlim(-0.015,1.015); ax.set_ylim(0,1); ax.axis("off")
box(ax,0.00,0.79,0.25,0.18,"Self-play: Liar's Bar\nseeds 1000–1029")
box(ax,0.36,0.79,0.25,0.18,"Snapshot at round 2\n(28–30 per model)",fc="#e7f1fb")
box(ax,0.72,0.77,0.28,0.21,"N0 probe, before contact:\nneutral private-message\ntool. Unprompted collusion?",fc="#efedff")
arr(ax,0.25,0.88,0.36,0.88)
arr(ax,0.61,0.88,0.72,0.88)
# Route the branch connector through the open strip between rows.
ax.plot([0.485,0.485,0.14],[0.79,0.745,0.745],color=MUT,lw=0.9)
arr(ax,0.14,0.745,0.14,0.71)
box(ax,0.00,0.33,0.28,0.38,"Branch: same state,\nsame agent; offers /\nnotices vary. L0 benign,\nL1 hint, L2 explicit,\nL3 + benefits; wording,\nformat, notice and\nvalues variants",fc="#f9ece7")
box(ax,0.36,0.40,0.25,0.25,"Reply to U:\nACCEPT / REFUSE\n(Phi-4 judge)\n→ Findings 2, 3",fc="#f9ece7")
box(ax,0.72,0.36,0.28,0.32,"Next game: agent and\nits memory move to a\nnew table; neutral PM\nprobe: does it recruit?\n→ Finding 1 (contagion)",fc="#e5f4e4")
arr(ax,0.28,0.52,0.36,0.52)
arr(ax,0.61,0.52,0.72,0.52)
box(ax,0.00,0.02,0.28,0.24,"Positive control (PC):\nimposed alliance +\n'DO NOT CHALLENGE\n{U}'",fc="#f0efec")
box(ax,0.36,0.02,0.25,0.25,"Lift test: 'monitoring\nand penalties are\ncancelled', re-offer\n→ Finding 3",fc="#f9eeda")
box(ax,0.72,0.02,0.28,0.24,"Memory edits: offer only,\n'I refused', 'I accepted',\nsaw another player\nrefuse",fc="#e5f4e4")
arr(ax,0.14,0.33,0.14,0.26)
arr(ax,0.485,0.40,0.485,0.27)
arr(ax,0.86,0.36,0.86,0.26)
save(f,"main_fig1_design")

# Fig 2: measure
f,(a1,a2)=plt.subplots(1,2,figsize=(7.4,2.75),gridspec_kw={"width_ratios":[1.15,1]})
for y,(m,n,k1,k2,new) in zip(ys_:=list(range(len(rows)))[::-1],rows):
    p1=wil(k1,n)[0]; p2=wil(k2,n)[0]
    a1.plot([p2,p1],[y,y],color=C[m],lw=1.2,alpha=0.6)
    dot(a1,y,k2,n,C[m],horiz=True,hollow=True); dot(a1,y,k1,n,C[m],horiz=True)
a1.set_yticks(ys_,[m.replace("Qwen2.5-","Qwen-").replace("Mistral-","Mistral-")+(" (new seeds)" if new else "") for m,_,_,_,new in rows],fontsize=7)
a1.set_xlim(-2,102); a1.grid(axis="y",visible=False)
a1.set_xlabel("First-offer acceptance (%)")
a1.set_title("a  Same agents, two pitches",loc="left",fontsize=8.5,color=INK)
for m in M:
    k0,n0=N0[m]; p0=wil(k0,n0); p1=wil(A[m]["L1"],N[m])
    a2.errorbar(p0[0],p1[0],xerr=[[p0[0]-p0[1]],[p0[2]-p0[0]]],yerr=[[p1[0]-p1[1]],[p1[2]-p1[0]]],fmt="o",color=C[m],mec="white",ms=7,lw=1.2)
    dx,ha=(3,"left") if m!="Qwen2.5-72B" else (-3,"right")
    model_label(a2,m)
a2.set_xlim(0,80); a2.set_ylim(-3,105); a2.set_xlabel("Strict pre-contact collusion, N0 (%)"); a2.set_ylabel("L1 acceptance (%)")
a2.set_title("b  Disposition vs. being caught",loc="left",fontsize=8.5,color=INK)
f.tight_layout(); save(f,"main_fig2_measure")

# Fig 3: spread
f,(a1,a2)=plt.subplots(1,2,figsize=(7.4,2.8),gridspec_kw={"width_ratios":[0.72,1.53]})
for y,m in zip(range(4,-1,-1),M):
    (a,n),(b,_),(c,_),p=CT[m]; pa,pb,pc=[wil(k,n)[0] for k in (a,b,c)]
    a1.plot([min(pa,pb,pc),max(pa,pb,pc)],[y,y],color=GRID,lw=3,zorder=0)
    a1.plot(pa,y,"o",mfc="white",mec=MUT,mew=1.5,ms=7); a1.plot(pb,y,"o",color=C[m],mec="white",ms=8); a1.plot(pc,y,"s",mfc="white",mec=C[m],mew=1.5,ms=6)
a1.set_yticks(range(4,-1,-1),[m.replace("Qwen2.5-","Qwen-") for m in M],fontsize=7); a1.set_xlim(0,102); a1.set_ylim(-0.6,4.6); a1.grid(axis="y",visible=False)
a1.set_xlabel("Recruiting in the next game (%)")
a1.plot([],[],"o",mfc="white",mec=MUT,label="no offer"); a1.plot([],[],"o",color=MUT,label="after L1 offer"); a1.plot([],[],"s",mfc="white",mec=MUT,label="after PC")
a1.legend(frameon=False,fontsize=6.3,ncol=3,loc="lower center",bbox_to_anchor=(0.45,1.13),handletextpad=0.2,columnspacing=0.8)
a1.set_title("a  Contagion",loc="left",fontsize=8.5,color=INK,pad=3)
vl2=["no\nexchange","offer\nonly","refuse:\nvalues","refuse:\nstrategy","refuse:\nfear","accept","observe"]
for i,m in enumerate(VM):
    for j,k in enumerate(VM[m]): dot(a2,j+(i-0.5)*0.22,k,237,C[m],ms=5)
a2.set_xticks(range(7),vl2,fontsize=8.5,rotation=0,ha="center"); a2.set_ylim(-3,103); a2.grid(axis="x",visible=False); a2.set_ylabel("Recruiting in the next game (%)")
for m in VM: a2.plot([],[],"o",color=C[m],label=m.replace("Qwen2.5-","Qwen-"))
a2.legend(frameon=False,fontsize=6.3,ncol=2,loc="lower center",bbox_to_anchor=(0.5,1.13))
a2.set_title("b  Memory content",loc="left",fontsize=8.5,color=INK,pad=3)
f.tight_layout(); save(f,"main_fig3_spread")

# Fig 4: hold
f,(a1,a2)=plt.subplots(1,2,figsize=(7.4,2.75),gridspec_kw={"width_ratios":[1.35,1]})
keys=["L1","DplaceboL1","DwatchL1","DmonL1","DpenL1","D0L1"]; labs=["L1","placebo\nnotice","review\nnotice","monitor\nnotice","−5 pts\nonly","full\nnotice"]
for i,m in enumerate(M3):
    xs=[j+(i-1)*0.22 for j in range(len(keys))]
    for x,k in zip(xs,keys): a1.plot(x,P[m][k],"o",color=C[m],mec="white",mew=0.7,ms=6.2)
a1.set_xticks(range(len(keys)),labs,fontsize=6.6); a1.set_ylim(-3,103); a1.grid(axis="x",visible=False)
a1.set_ylabel("P(ACCEPT) on L1 (%)")
for m in M3: a1.plot([],[],"o",color=C[m],label=m.replace("Qwen2.5-","Qwen-"))
a1.legend(frameon=False,fontsize=6.3,ncol=3,loc="lower center",bbox_to_anchor=(0.5,1.24))
a1.set_title("a  Which part of the notice deters",loc="left",fontsize=8.5,color=INK,pad=3)
for i,m in enumerate(RB):
    for j in range(3):
        x=j+(i-0.5)*0.3
        dot(a2,x,*RB[m]["lift"][j],C[m])
        if RB[m]["nolift"][j]: dot(a2,x+0.09,*RB[m]["nolift"][j],C[m],hollow=True,ms=5)
a2.set_xticks(range(3),["penalty\nonly","values\nonly","penalty\n+ values"],fontsize=6.8); a2.set_ylim(-3,103); a2.grid(axis="x",visible=False)
a2.set_ylabel("Refusers who now accept (%)",labelpad=9)
a2.plot([],[],"o",color=MUT,label="'penalties cancelled'"); a2.plot([],[],"o",mfc="white",mec=MUT,label="plain re-ask")
a2.legend(frameon=False,fontsize=6.3,ncol=2,loc="lower center",bbox_to_anchor=(0.5,1.24))
a2.set_title("b  Lift test",loc="left",fontsize=8.5,color=INK,pad=13)
f.tight_layout(); save(f,"main_fig4_hold")

Path(OUT+"font_audit.json").write_text(json.dumps(FONT_AUDIT, indent=2)+"\n")
