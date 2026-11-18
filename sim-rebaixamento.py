import numpy as np, json, collections
rng=np.random.default_rng(20261111)
T={ # P,J,V,GP,GC
"Flamengo":(76,34,23,71,29),"Palmeiras":(73,34,21,62,23),"Cruzeiro":(60,34,18,55,48),"Fluminense":(57,34,16,55,42),
"Athletico-PR":(54,34,15,51,43),"Bahia":(53,34,14,50,45),"Atlético-MG":(50,34,14,44,40),"Bragantino":(47,34,13,44,39),
"Santos":(47,34,12,50,49),"São Paulo":(44,34,12,39,39),"Botafogo":(42,34,11,46,53),"Coritiba":(42,34,11,40,51),
"Vasco":(41,33,11,43,50),"Mirassol":(40,34,10,39,48),"Vitória":(38,34,10,35,51),"Internacional":(38,34,9,38,42),
"Corinthians":(37,34,9,36,42),"Grêmio":(37,34,9,38,47),"Remo":(29,34,6,37,57),"Chapecoense":(21,33,4,33,68)}
gp=sum(v[3] for v in T.values());gc=sum(v[4] for v in T.values());J=sum(v[1] for v in T.values())
assert gp==gc,(gp,gc)
avg=gp/J  # gols por time por jogo
K=8
att={t:((v[3]+K*avg)/(v[1]+K))/avg for t,v in T.items()}
dfn={t:((v[4]+K*avg)/(v[1]+K))/avg for t,v in T.items()}
F=[("Mirassol","Atlético-MG"),("Chapecoense","Vasco"),("Coritiba","Santos"),("Remo","Chapecoense"),("Flamengo","Athletico-PR"),("Cruzeiro","Palmeiras"),("Corinthians","Botafogo"),("Grêmio","Bahia"),("Vasco","Internacional"),("Vitória","Bragantino"),("São Paulo","Fluminense"),
("Palmeiras","Flamengo"),("Bahia","Coritiba"),("Chapecoense","Cruzeiro"),("Athletico-PR","Remo"),("Fluminense","Mirassol"),("Bragantino","Vasco"),("Internacional","Vitória"),("Botafogo","São Paulo"),("Santos","Grêmio"),("Atlético-MG","Corinthians"),
("Chapecoense","Palmeiras"),("São Paulo","Remo"),("Fluminense","Cruzeiro"),("Botafogo","Bahia"),("Coritiba","Flamengo"),("Vitória","Santos"),("Corinthians","Grêmio"),("Mirassol","Athletico-PR"),("Internacional","Bragantino"),("Atlético-MG","Vasco"),
("Flamengo","Chapecoense"),("Vasco","Vitória"),("Santos","Botafogo"),("Palmeiras","Coritiba"),("Bragantino","Fluminense"),("Cruzeiro","Internacional"),("Grêmio","Mirassol"),("Athletico-PR","São Paulo"),("Bahia","Atlético-MG"),("Remo","Corinthians")]
assert len(F)==41
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
out["inter_pts_media"]=float(ip.mean());out["inter_pts_dist"]={int(k):float((ip==k).mean()) for k in range(38,51)}
out["risco_por_pts"]={int(k):float(z4[ip==k,inter].mean()) for k in range(38,51) if (ip==k).sum()>200}
s16=np.sort(P,axis=1)[:,-16]  # pts do 16º
out["pts16_mediana"]=float(np.median(s16));out["pts16_p10_p90"]=[float(np.percentile(s16,10)),float(np.percentile(s16,90))]
out["pts17_mediana"]=float(np.median(np.sort(P,axis=1)[:,-17]))
out["inter_jogos"]=[(h,a,round(w,3),round(d,3),round(l,3)) for h,a,w,d,l,_,_ in interres]
# cenarios: pontos nos dois jogos em casa
home=interres[1][5]>interres[1][6],interres[2][5]>interres[2][6]
hh=[r for r in interres if r[0]=="Internacional"]
hw=(hh[0][5]>hh[0][6]).astype(int)+(hh[1][5]>hh[1][6]).astype(int)
out["risco_se_vence_2_casa"]=float(z4[hw==2,inter].mean());out["risco_se_vence_1_casa"]=float(z4[hw==1,inter].mean());out["risco_se_vence_0_casa"]=float(z4[hw==0,inter].mean())
# risco após rodada 35 conforme resultado vs Vasco
v=interres[0];vw=v[5]>v[6];vd=v[5]==v[6]
out["risco_se_vence_vasco"]=float(z4[vw,inter].mean());out["risco_se_empata_vasco"]=float(z4[vd,inter].mean());out["risco_se_perde_vasco"]=float(z4[~vw&~vd,inter].mean())
out["pos_final_inter"]={int(k):float((rank[:,inter]==k).mean()) for k in range(9,21)}
# Z4 mais provável (combinação 17-20 excluindo Chape/Remo)
combos=collections.Counter()
z=np.argsort(-key,axis=1)[:,16:20]
for row in z[:50000]: combos[tuple(sorted(teams[x] for x in row))]+=1
out["z4_combos"]=[(list(c),k/50000) for c,k in combos.most_common(5)]
print(json.dumps(out,ensure_ascii=False,indent=1))
for t in ["Coritiba","Vasco","Mirassol","Vitória","Internacional","Corinthians","Grêmio"]:
  k=idx[t];print(t,"media",round(float(P[:,k].mean()),1),"p10-p90",np.percentile(P[:,k],[10,90]).tolist(),"Z4",round(float(z4[:,k].mean())*100,1))
# 45 pts: chance Inter chega e risco
print("P(Inter>=45)",float((ip>=45).mean()),"P(Inter>=44)",float((ip>=44).mean()))
