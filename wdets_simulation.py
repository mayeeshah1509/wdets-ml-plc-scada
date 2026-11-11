"""
WDETS: ML-driven PLC controller for a wireless power transfer (WPT) link with SCADA,
power-factor correction and PID control. Reproduces every table in the paper.
Run:  python3 wdets_simulation.py     (Python 3, numpy, scikit-learn)  seed = 42
"""
import numpy as np, math, csv, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
rng = np.random.default_rng(42)

# ---------------- PART A: resonant WPT link (series-series compensation) ----------------
f0=85e3; w=2*math.pi*f0; L1=L2=40e-6; R1=R2=0.15; RL=10.0
C1=C2=1/(w**2*L1); Rcoil=0.15      # coil radius (m)
def k_of_d(d): return 0.5*(1+(d/Rcoil)**2)**-1.5   # geometry factor 0.5 (assumption)
def link(d, delta=0.0, C1f=1.0):
    k=k_of_d(d); M=k*math.sqrt(L1*L2)
    Z2=R2+RL                                   # receiver tuned
    Zr=(w*M)**2/Z2
    X1=w*L1*(1+delta)-1/(w*C1*C1f)
    Zin=R1+Zr+1j*X1
    I1=1/Zin; I2=abs(1j*w*M*I1/Z2)
    Pin=(abs(I1)**2)*(R1+Zr); PL=I2**2*RL
    return k, PL/Pin, math.cos(np.angle(Zin)), abs(I1)
Q1=w*L1/R1
rowsA=[]
print("A1 efficiency vs distance")
for d in [0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30]:
    k,eta,pf,I=link(d); x=k*k*Q1*Q1; emax=x/(1+math.sqrt(1+x))**2
    rowsA.append((d,k,eta,emax)); print(round(d*100),round(k,3),round(eta*100,1),round(emax*100,1))
print("A2 detuning and PLC capacitor retune (d=10 cm)")
steps=np.arange(-20,21)*0.005     # 41 steps of 0.5 % of C1 (+-10 %)
def best_step(d,delta):
    best=max(steps,key=lambda s: link(d,delta,1+s)[2]); return best
# dataset for ML: features measured by PLC (phase angle, |I1|), target best step
def make(n):
    X=[];y=[];meta=[]
    for _ in range(n):
        d=rng.uniform(0.06,0.20); delta=rng.uniform(-0.08,0.08)
        k,eta,pf,I=link(d,delta); ph=math.acos(pf)*np.sign(delta if delta!=0 else 1)
        # signed phase from X1 sign
        Zin_ph=np.angle(R1+(w*k*math.sqrt(L1*L2))**2/(R2+RL)+1j*(w*L1*delta))
        X.append([Zin_ph,I]);y.append(best_step(d,delta));meta.append((d,delta))
    return np.array(X),np.array(y),meta
Xtr,ytr,_=make(1500); Xte,yte,mte=make(500)
sc=StandardScaler().fit(Xtr)
gb=GradientBoostingRegressor(random_state=42).fit(sc.transform(Xtr),ytr)
pred=gb.predict(sc.transform(Xte))
snap=lambda p: steps[np.argmin(abs(steps-p))]
pf0=[];pf1=[];e0=[];e1=[];ok=0
for (d,delta),p,t in zip(mte,pred,yte):
    s=snap(p); a=link(d,delta,1.0); b=link(d,delta,1+s)
    pf0.append(a[2]);e0.append(a[1]);pf1.append(b[2]);e1.append(b[1]); ok+= abs(s-t)<=0.0051
print("PF before",np.mean(pf0),"PF after",np.mean(pf1),"eta before",np.mean(e0),"eta after",np.mean(e1),"within 1 step",ok/len(yte))
print("min PF before",min(pf0),"min PF after",min(pf1))
A2=dict(pf0=np.mean(pf0),pf1=np.mean(pf1),e0=np.mean(e0),e1=np.mean(e1),ok=ok/len(yte),mn0=min(pf0),mn1=min(pf1))
# worked example
for delta in [0.0,0.03,0.06]:
    a=link(0.10,delta,1.0); s=best_step(0.10,delta); b=link(0.10,delta,1+s)
    print("example delta",delta,"PF",round(a[2],3),"eta",round(a[1]*100,1),"-> step",round(s*100,1),"%  PF",round(b[2],3),"eta",round(b[1]*100,1))

# ---------------- PART B: site power-factor control (SCADA 5-min records) ----------------
days=30; N=days*288; t=np.arange(N); hr=(t%288)/12.0; dow=(t//288)%7
def ar(n,phi,s):
    e=rng.normal(0,s,n);x=np.zeros(n)
    for i in range(1,n): x[i]=phi*x[i-1]+e[i]
    return x
work=(dow<5).astype(float)
P=450+220*np.sin(2*math.pi*(hr-9)/24)*1.0+120*work*((hr>8)&(hr<18))+ar(N,0.9,14)
ev=np.zeros(N)
for _ in range(int(N/40)):   # motor start/stop events
    i=rng.integers(0,N-12); ev[i:i+rng.integers(3,12)]+=rng.choice([-60,70,90])
Q=0.80*P+50*np.sin(2*math.pi*(hr-3)/24)+ar(N,0.85,16)+ev
P=np.clip(P,80,None); Q=np.clip(Q,0,None)
BANK=50.0; NST=8; TAR=math.tan(math.acos(0.98))
ntr=20*288
def feats(i):
    return [Q[i-j] for j in range(1,7)]+[P[i-j] for j in range(1,7)]+[math.sin(2*math.pi*hr[i]/24),math.cos(2*math.pi*hr[i]/24),work[i]]
idx=np.arange(6,N); X=np.array([feats(i) for i in idx]); 
trm=idx<ntr; tem=~trm
scB=StandardScaler().fit(X[trm]); Xs=scB.transform(X)
mods={}
for name,mk in [("ridge",lambda: Ridge(1.0)),("gbr",lambda: GradientBoostingRegressor(random_state=42))]:
    mq=mk().fit(Xs[trm],Q[idx][trm]); mp=mk().fit(Xs[trm],P[idx][trm])
    mods[name]=(mq.predict(Xs[tem]),mp.predict(Xs[tem]))
ti=idx[tem]; Pt=P[ti]; Qt=Q[ti]; Qlag=Q[ti-1]; Plag=P[ti-1]
def nsteps(q,p): return np.clip(np.round((q-TAR*p)/BANK),0,NST)
def evalS(n):
    qn=Qt-n*BANK; S=np.sqrt(Pt**2+qn**2); pf=Pt/S
    sw=np.sum(abs(np.diff(n))); loss=np.sum(Pt**2+qn**2)/np.sum(Pt**2+Qt**2)
    return dict(pf=pf.mean(),pfE=Pt.sum()/S.sum(),low=(pf<0.95).mean()*100,sw=sw/len(n)*288,loss=loss*100)
res={}
res["No compensation"]=evalS(np.zeros(len(ti)))
res["SCADA-reactive (uses last scan)"]=evalS(nsteps(Qlag,Plag))
res["ML-PLC ridge forecast"]=evalS(nsteps(*mods["ridge"]))
res["ML-PLC gradient-boosting forecast"]=evalS(nsteps(mods["gbr"][0],mods["gbr"][1]))
res["Oracle (perfect knowledge)"]=evalS(nsteps(Qt,Pt))
print("B site PF (test = last 10 days, n=%d)"%len(ti))
for k,v in res.items(): print(k,{a:round(b,3) for a,b in v.items()})
fq=np.sqrt(np.mean((mods["ridge"][0]-Qt)**2)); print("Q forecast RMSE ridge",fq,"naive",np.sqrt(np.mean((Qlag-Qt)**2)),"gbr",np.sqrt(np.mean((mods["gbr"][0]-Qt)**2)))
print("raw PF mean uncompensated train+test",np.mean(P/np.sqrt(P**2+Q**2)))
with open("site_scada_5min.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["t_5min","hour","weekday","P_kW","Q_kVAR","PF"])
    for i in range(N): wr.writerow([i,round(hr[i],2),int(dow[i]),round(P[i],1),round(Q[i],1),round(P[i]/math.hypot(P[i],Q[i]),4)])

# ---------------- PART C: PID on output-voltage loop, PLC scan 10 ms, gain scheduling ----------------
wn=30.0; zt=0.3; Ts=0.01; dt=0.002; sub=int(Ts/dt); T=3.0
def sim(g,Kp,Ki,Kd,beta=1.0,noise=0.0,seed=0,Tm=None):
    r=np.random.default_rng(seed); n=int((Tm or T)/dt); x=0.0; v=0.0; I=0.0; df=0.0; yprev=0.0; u=0.0; ys=np.zeros(n); N_=15
    ypend=0.0
    for k in range(n):
        if k%sub==0:
            ym=x+noise*r.normal()
            e=1-ym; I+=Ki*e*Ts
            dd=-(ym-yprev)/Ts; yprev=ym; df+=(dd-df)*Ts*N_
            uu=Kp*(beta-ym)+I+Kd*df; us=min(max(uu,0),4)
            I-=(uu-us)*0.5; u=us
        a=g*wn*wn*u-2*zt*wn*v-wn*wn*x; v+=a*dt; x+=v*dt
        ys[k]=x
        if abs(x)>50: ys[k:]=50; break
    return ys
def met(y):
    os=max(0,(y.max()-1)*100); idx=np.where(abs(y-1)>0.02)[0]; ts=(idx[-1]+1)*dt if len(idx) else 0
    ise=np.sum((1-y)**2)*dt; return os,ts,ise,abs(1-y[-1])*100
def tune(g,trials=500,seed=1):
    r=np.random.default_rng(seed); best=None
    for _ in range(trials):
        Kp=r.uniform(0.05,3); Ki=r.uniform(0,12); Kd=r.uniform(0,0.15)
        y=sim(g,Kp,Ki,Kd); os,ts,ise,ess=met(y)
        J=ise+0.02*os+0.15*ts+0.5*ess
        if best is None or J<best[0]: best=(J,(Kp,Ki,Kd))
    return best[1]
kn=k_of_d(0.10)
gof=lambda d: kn/k_of_d(d)
dtr=[0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20]; dte=[0.07,0.09,0.11,0.13,0.15,0.17,0.19]
gtr=[gof(d) for d in dtr]
gains_tr=[tune(g,seed=i+1) for i,g in enumerate(gtr)]
print("C training gains"); 
for d,g,k_ in zip(dtr,gtr,gains_tr): print(d,round(g,2),[round(a,3) for a in k_])
G=np.array(gains_tr); lg=np.log(gtr)
coef=[np.polyfit(lg,G[:,j],2) for j in range(3)]
sched=lambda g: [max(0,np.polyval(c,math.log(g))) for c in coef]
fixed=gains_tr[dtr.index(0.10)]
print("fixed gains (tuned at 10 cm)",fixed)
rowsC=[]
print("C test: d, g, fixed(OS,Ts,ISE) | ML-sched | oracle")
for i,d in enumerate(dte):
    g=gof(d); mf=met(sim(g,*fixed)); ms=met(sim(g,*sched(g))); orc=tune(g,seed=100+i); mo=met(sim(g,*orc))
    rowsC.append((d,g,mf,ms,mo)); print(round(d*100),round(g,2),[round(a,2) for a in mf],[round(a,2) for a in ms],[round(a,2) for a in mo])
print("avg fixed",np.mean([r[2][0] for r in rowsC]),np.mean([r[2][1] for r in rowsC]),"worst OS fixed",max(r[2][0] for r in rowsC))
print("avg sched",np.mean([r[3][0] for r in rowsC]),np.mean([r[3][1] for r in rowsC]),"worst OS sched",max(r[3][0] for r in rowsC))
print("avg oracle",np.mean([r[4][0] for r in rowsC]),np.mean([r[4][1] for r in rowsC]))
# noise robustness
d=0.15; g=gof(d)
print("noise 0.5% fixed",met(sim(g,*fixed,noise=0.005,seed=3)),"sched",met(sim(g,*sched(g),noise=0.005,seed=3)))
# P-only / PI / PID / PID+filter reference at nominal
print("nominal plant: open-loop OS", max(0,(np.max(np.array([1-math.exp(-zt*wn*tt)*(math.cos(wn*math.sqrt(1-zt**2)*tt)+zt/math.sqrt(1-zt**2)*math.sin(wn*math.sqrt(1-zt**2)*tt)) for tt in np.arange(0,T,dt)]))-1)*100))
print("P",met(sim(1,1.0,0,0)),"PI",met(sim(1,0.8,6,0)),"PID",met(sim(1,*fixed)))
# margins with 1.5 scan delay
def margins(g,Kp,Ki,Kd):
    ww=np.logspace(-1,3,20000); s=1j*ww
    Gs=g*wn**2/(s**2+2*zt*wn*s+wn**2); C=Kp+Ki/s+Kd*s/(1+s/15.0); L=Gs*C*np.exp(-s*0.5*Ts)
    mag=abs(L); ph=np.unwrap(np.angle(L))
    ic=np.where(np.diff(np.sign(mag-1)))[0]; pm=180+np.degrees(ph[ic[0]]) if len(ic) else float('nan')
    ip=np.where(np.diff(np.sign(ph+math.pi)))[0]; gm=20*math.log10(1/mag[ip[0]]) if len(ip) else float('inf')
    return pm,gm
print("margins (derivative filter pole 15 rad/s, ZOH delay 0.5 scan)")
for d in [0.07,0.11,0.15,0.19]:
    g=gof(d); print(d,round(g,2),"fixed",[round(a,1) for a in margins(g,*fixed)],"sched",[round(a,1) for a in margins(g,*sched(g))])
def routh(g,Kp,Ki,Kd): return (2*zt*wn+g*wn**2*Kd)*(wn**2*(1+g*Kp))>g*wn**2*Ki
print("routh all test sched",all(routh(gof(d),*sched(gof(d))) for d in dte),"fixed",[routh(gof(d),*fixed) for d in dte])
# SCADA data volume
print("SCADA 24 tags @1s per day",24*86400)

# ---------------- PART D: packet-style ("electricity like data") delivery ----------------
# Each packet = one receiver, one requested power level (reference 1 pu) held for Tp seconds.
# Receivers sit at different distances, so the plant gain g changes from packet to packet.
# Metrics per packet: usable fraction (time within +-2 % of requested amplitude),
# energy error (delivered vs requested, power ~ y^2) and peak power overshoot.
print("D packets")
rowsD=[]
for Tp in [1,2,5,10]:
    res={"fixed":[],"sched":[]}
    for d in dte:
        g=gof(d)
        for name,gains in [("fixed",fixed),("sched",sched(g))]:
            y=sim(g,*gains,Tm=Tp)
            usable=np.mean(abs(y-1)<0.02)*100
            eerr=(np.sum(y**2)*dt/Tp-1)*100
            pk=max(0,(y.max()**2-1)*100)
            res[name].append((usable,abs(eerr),pk))
    out=[]
    for name in ["fixed","sched"]:
        a=np.array(res[name]); out+= [a[:,0].mean(),a[:,0].min(),a[:,1].mean(),a[:,2].max()]
    rowsD.append((Tp,out)); print(Tp,[round(v,2) for v in out])
