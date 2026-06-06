import struct, sys
f=open(sys.argv[1],'rb')
def rd(n): return f.read(n)
def u32(): return struct.unpack('<I',rd(4))[0]
def u64(): return struct.unpack('<Q',rd(8))[0]
def i32(): return struct.unpack('<i',rd(4))[0]
def i64(): return struct.unpack('<q',rd(8))[0]
def f32(): return struct.unpack('<f',rd(4))[0]
def f64(): return struct.unpack('<d',rd(8))[0]
def b(): return rd(1)[0]!=0
def s():
    n=u64(); return rd(n).decode('utf-8','replace')
def val(t):
    if t==0: return b'\x00'==rd(1) and 0 or 0  # uint8 (read 1)
    if t==1: return struct.unpack('<b',rd(1))[0]
    if t==2: return struct.unpack('<H',rd(2))[0]
    if t==3: return struct.unpack('<h',rd(2))[0]
    if t==4: return u32()
    if t==5: return i32()
    if t==6: return f32()
    if t==7: return b()
    if t==8: return s()
    if t==10: return u64()
    if t==11: return i64()
    if t==12: return f64()
    if t==9:
        et=u32(); cnt=u64(); return [val(et) for _ in range(cnt)]
    raise ValueError(f"type {t}")
assert rd(4)==b'GGUF'
ver=u32(); tc=u64(); kvc=u64()
print("gguf_version",ver,"tensors",tc,"kv_count",kvc)
want=("general.architecture","general.name",".context_length",".block_count","tokenizer.chat_template","tokenizer.ggml.model")
for _ in range(kvc):
    k=s(); t=u32(); v=val(t)
    if any(w in k for w in want) or k.endswith("context_length"):
        if k=="tokenizer.chat_template":
            v=str(v); print("KEY",k,"len",len(v),"| has<think>:", "<think>" in v, "| enable_thinking:", "enable_thinking" in v, "| /think:", "/think" in v)
            print("  template_head:", v[:300].replace("\n"," "))
        else:
            print("KEY",k,"=",v)
