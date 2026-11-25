import numpy as np, json, collections
rng=np.random.default_rng(20261125)
T={ # P,J,V,GP,GC  (tabela valorfinal.com.br atualizada 25/11/2026 00:00, apos 36a rodada + Chapecoense 3x0 Palmeiras da 37a; Atletico-MG x Corinthians pendente 25/11 20h)
"Flamengo":(80,36,24,74,31),"Palmeiras":(74,37,21,64,29),"Cruzeiro":(66,36,20,59,50),"Fluminense":(63,36,18,57,42),
"Athletico-PR":(57,36,16,56,47),"Bahia":(54,36,14,50,46),"Atlético-MG":(51,35,14,44,40),"Bragantino":(50,36,14,46,41),
"Santos":(50,36,13,51,51),"São Paulo":(45,36,12,40,41),"Botafogo":(43,36,11,47,58),"Coritiba":(45,36,12,41,52),
"Vasco":(47,36,13,47,52),"Mirassol":(41,36,10,39,49),"Vitória":(44,36,12,42,54),"Internacional":(38,36,9,41,49),
"Corinthians":(40,35,10,40,42),"Grêmio":(41,36,10,40,47),"Remo":(32,36,7,41,61),"Chapecoense":(24,37,5,37,74)}
gp=sum(v[3] for v in T.values());gc=sum(v[4] for v in T.values());J=sum(v[1] for v in T.values())
assert gp==gc,(gp,gc)
avg=gp/J  # gols por time por jogo
K=8
att={t:((v[3]+K*avg)/(v[1]+K))/avg for t,v in T.items()}
dfn={t:((v[4]+K*avg)/(v[1]+K))/avg for t,v in T.items()}
F=[
("Atlético-MG","Corinthians"),
("São Paulo","Remo"),("Fluminense","Cruzeiro"),("Botafogo","Bahia"),("Coritiba","Flamengo"),("Vitória","Santos"),("Corinthians","Grêmio"),("Internacional","Bragantino"),("Mirassol","Athletico-PR"),("Atlético-MG","Vasco"),
("Flamengo","Chapecoense"),("Vasco","Vitória"),("Santos","Botafogo"),("Palmeiras","Coritiba"),("Bragantino","Fluminense"),("Cruzeiro","Internacional"),("Grêmio","Mirassol"),("Athletico-PR","São Paulo"),("Bahia","Atlético-MG"),("Remo","Corinthians")]
assert len(F)==20
cnt=collections.Counter(); 
for h,a in F: cnt[h]+=1;cnt[a]+=1
for t,v in T.items(): assert v[1]+cnt[t]==38,(t,v[1],cnt[t])
teams=list(T);idx={t:i for i,t in enumerate(teams)};n=len(teams)
N=200000;HA=1.16;AA=0.86
P0=np.array([T[t][0] for t in teams]);W0=np.array([T[t][2] for t in teams]);GD0=np.array([T[t][3]-T[t][4] for t in teams]);GF0=np.array([T[t][3] for t in teams])
# incerteza de força: ruído lognormal por simulação
na=rng.lognormal(0,0.12,(N,n));nd=rng.lognormal(0,0.12,(N,n))
A=np.array([att[t] for t in teams])*na;D=np.array([dfn[t] for t in teams])*nd
P=np.tile(P0,(N,1)).astype(float);W=np.tile(W0,(N,1));GD=np.tile(GD0,(N,1));GF=np.tile(GF0,(N,1))
inter=idx["Internacional"];interres=[]
for h,a in F:
  i,j=idx[h],idx[a]
  gh=rng.poisson(avg*HA*A[:,i]*D[:,j]);ga=rng.poisson(avg*AA*A[:,j]*D[:,i])
  P[:,i]+=np.where(gh>ga,3,np.where(gh==ga,1,0));P[:,j]+=np.where(ga>gh,3,np.where(gh==ga,1,0))
  W[:,i]+=gh>ga;W[:,j]+=ga>gh;GD[:,i]+=gh-ga;GD[:,j]+=ga-gh;GF[:,i]+=gh;GF[:,j]+=ga
  if inter in(i,j):
    me,ot=(gh,ga) if i==inter else (ga,gh)
    interres.append((h,a,(me>ot).mean(),(me==ot).mean(),(me<ot).mean(),me,ot))
key=P*1e9+W*1e6+(GD+500)*1e3+GF+rng.random((N,n))*1e-3
rank=np.argsort(np.argsort(-key,axis=1),axis=1)+1
z4=(rank>=17)
out={"avg":avg}
out["risco"]={t:float(z4[:,idx[t]].mean()) for t in teams if T[t][0]<=47}
ip=P[:,inter]
out["inter_pts_media"]=float(ip.mean());out["inter_pts_dist"]={int(k):float((ip==k).mean()) for k in range(38,47)}
out["risco_por_pts"]={int(k):float(z4[ip==k,inter].mean()) for k in range(38,47) if (ip==k).sum()>200}
s16=np.sort(P,axis=1)[:,-16]  # pts do 16º
out["pts16_mediana"]=float(np.median(s16));out["pts16_p10_p90"]=[float(np.percentile(s16,10)),float(np.percentile(s16,90))]
out["pts17_mediana"]=float(np.median(np.sort(P,axis=1)[:,-17]))
out["inter_jogos"]=[(h,a,round(w,3),round(d,3),round(l,3)) for h,a,w,d,l,_,_ in interres]
# cenarios: dois jogos restantes (Bragantino em casa, Cruzeiro fora)
pr=[(r[5]>r[6]).astype(int)*3+(r[5]==r[6]).astype(int) for r in interres]
tot=pr[0]+pr[1]
out["risco_por_pts_dois_jogos"]={int(k):float(z4[tot==k,inter].mean()) for k in range(0,7) if (tot==k).sum()>0}
out["risco_se_vence_proximo"]=float(z4[pr[0]==3,inter].mean());out["risco_se_empata_proximo"]=float(z4[pr[0]==1,inter].mean());out["risco_se_perde_proximo"]=float(z4[pr[0]==0,inter].mean())
for nm,ix in (("vence",pr[0]==3),("empata",pr[0]==1),("perde",pr[0]==0)):
  for nm2,iy in (("vence",pr[1]==3),("empata",pr[1]==1),("perde",pr[1]==0)):
    out["risco_%s_bragantino_%s_cruzeiro"%(nm,nm2)]=float(z4[ix&iy,inter].mean())
out["pos_final_inter"]={int(k):float((rank[:,inter]==k).mean()) for k in range(9,21)}
# Z4 mais provável (combinação 17-20 excluindo Chape/Remo)
combos=collections.Counter()
z=np.argsort(-key,axis=1)[:,16:20]
for row in z[:50000]: combos[tuple(sorted(teams[x] for x in row))]+=1
out["z4_combos"]=[(list(c),k/50000) for c,k in combos.most_common(5)]
print(json.dumps(out,ensure_ascii=False,indent=1))
for t in ["Coritiba","Vasco","Mirassol","Vitória","Internacional","Corinthians","Grêmio","Botafogo"]:
  k=idx[t];print(t,"media",round(float(P[:,k].mean()),1),"p10-p90",np.percentile(P[:,k],[10,90]).tolist(),"Z4",round(float(z4[:,k].mean())*100,1))
# 45 pts: chance Inter chega e risco
print("P(Inter>=44)",float((ip>=44).mean()))

print("--- PROJECAO")
for t in sorted(teams,key=lambda t:-P[:,idx[t]].mean()):
  k=idx[t];print(t,round(float(P[:,k].mean()),1),np.percentile(P[:,k],[10,90]).tolist(),"Z4",round(float(z4[:,k].mean())*100,1))
print("rank mais provavel Inter",max(out["pos_final_inter"].items(),key=lambda x:x[1]))
print("P(Inter 16o ou melhor)",float((rank[:,inter]<=16).mean()))
print("risco_por_pts",{k:round(v*100,1) for k,v in out["risco_por_pts"].items()})
print("dist pts",{k:round(v*100,1) for k,v in out["inter_pts_dist"].items()})
