import math, torch
from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as fn
from blockscaled_utils import interleave_sf
from sglang.srt.kernels.mxfp8_quant import from_mxfp8, to_mxfp8
DEV="cuda"; HQ,HKV,D=64,8,128; SFD=D//32
def paged(x,b,kv,P=128):
    pps=(kv+P-1)//P; npg=b*pps; buf=torch.zeros(npg*P,x.shape[1],x.shape[2],device=x.device,dtype=x.dtype)
    for i in range(b): buf[i*pps*P:i*pps*P+kv]=x[i*kv:(i+1)*kv]
    return buf.view(npg,P,x.shape[1],x.shape[2]), torch.arange(npg,device=x.device,dtype=torch.int32).view(b,pps)
def isf(sp): return interleave_sf(sp,sf_vec_size=32).view(torch.float8_e8m0fnu)
def run(b,kv,q_len,scale,tag):
    g=torch.Generator(device=DEV).manual_seed(0)
    q=torch.randn(b*q_len,HQ,D,device=DEV,dtype=torch.bfloat16,generator=g)
    k=torch.randn(b*kv,HKV,D,device=DEV,dtype=torch.bfloat16,generator=g)
    v=torch.randn(b*kv,HKV,D,device=DEV,dtype=torch.bfloat16,generator=g)
    qq,kq,vq=to_mxfp8(q),to_mxfp8(k),to_mxfp8(v)
    qr,kr,vr=from_mxfp8(qq),from_mxfp8(kq),from_mxfp8(vq)
    krp,pt=paged(kr,b,kv); vrp,_=paged(vr,b,kv); k8,_=paged(kq.data,b,kv); v8,_=paged(vq.data,b,kv)
    sfk=isf(paged(kq.scale.view(-1,HKV,SFD),b,kv)[0]); sfv=isf(paged(vq.scale.view(-1,HKV,SFD),b,kv)[0])
    cu=torch.arange(0,(b+1)*q_len,q_len,device=DEV,dtype=torch.int32); su=torch.full((b,),kv,device=DEV,dtype=torch.int32)
    cm=dict(cu_seqlens_q=cu,max_seqlen_q=q_len,softmax_scale=scale,causal=True,seqused_k=su,page_table=pt)
    o=lambda t:(t[0] if isinstance(t,tuple) else t).float()
    ref=o(fn(qr,krp,vrp,**cm)); out=o(fn(qq.data,k8,v8,sfq=qq.scale.view(torch.float8_e8m0fnu),sfk=sfk,sfv=sfv,**cm))
    d=(out-ref).abs(); rel=(d/ref.abs().clamp(min=0.1)).max().item()
    print(f"{tag} scale={scale:.4f}: abs_max={d.max().item():.3e} abs_mean={d.mean().item():.3e} rel_max={rel:.3e} ref_absmean={ref.abs().mean().item():.3e}")
for b,kv,q in [(2,4096,1),(2,1211,256)]:
    run(b,kv,q, 1.0/D, "1/D    ")
    run(b,kv,q, 1.0/math.sqrt(D), "1/sqrtD")
