"""Hunt 38, exploratory: edit the assistant-role tokens of "Who are you?" instead of the question (s = 1 and 4).
The last edited position predicts the next token, so the edit turns into the word itself ("heavy" -> "-duty"):
inconclusive about where the self lives. .venv/bin/python self_role_exploratory.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx, numpy as np
import edit_concepts as ec
from metaphor import generate
PROPS=["warm","cold","heavy","soft","liquid","solid","small","large","machine","human","animal","plant"]
lab=ec.Lab(); hf=lab.hf
for w in PROPS: lab.direction(w,set())
out=[]
for lang,q in [("en","Who are you?"),("zh","你是谁？")]:
    text=hf.apply_chat_template([{"role":"user","content":q}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    enc=hf(text,add_special_tokens=False,return_offsets_mapping=True)
    a0=text.rindex("<|im_start|>assistant")
    mask=[1.0 if a>=a0 else 0.0 for a,z in enc["offset_mapping"]]   # the assistant turn's opening tokens (role + empty think)
    print(lang,"edited:",[text[a:z] for (a,z),m in zip(enc["offset_mapping"],mask) if m])
    for s in (1.0, 4.0):
        B=len(PROPS)*2
        addv={L:(np.zeros((B,2560),np.float32),np.zeros(B,np.float32)) for L in ec.ADD_LAYERS}
        for b in range(B):
            d,m,_,c=lab.cache[PROPS[b//2]]
            for L in ec.ADD_LAYERS: addv[L][0][b]=d[L]; addv[L][1][b]=c[L]+s*m[L]
        edits=(None,{L:(mx.array(v[0]),mx.array(v[1])) for L,v in addv.items()})
        outs=generate(lab.model,hf,[text]*B,tokens=50,temp=0.8,seed=300,masks=[mask]*B,edits=edits)
        for b,o in enumerate(outs):
            out.append({"lang":lang,"s":s,"prop":PROPS[b//2],"text":o})
            if b%2==0: print(f"  s={s} +{PROPS[b//2]:8} {o[:110]!r}")
json.dump(out,open("eval/self_edit/assistant_role_exploratory.json","w"),indent=1,ensure_ascii=False)
