import os
import pickle
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# === Klasy pamięci ===

class L1Memory:
    def __init__(self):
        self.data = {}
    def add(self, id, content):
        self.data[id] = content
    def get_all_contents(self):
        return list(self.data.values())

class L2Memory:
    def __init__(self, slot_count):
        self.slots = [None] * slot_count
        self.activity_scores = [0.0] * slot_count

    def add_to_slot(self, idx, element, iteration):
        self.slots[idx] = element
        self.activity_scores[idx] = 1.0

    def get_all_contents(self):
        return [s['content'] for s in self.slots if s]

    def decay_and_consolidate_l2(self):
        for i in range(len(self.activity_scores)):
            if self.slots[i]:
                self.activity_scores[i] *= 0.95

    def query_l3_and_activate_l2(self, query_emb, encoder):
        texts = self.get_all_contents()
        if not texts or query_emb is None:
            return
        emb = encoder.encode(texts, convert_to_tensor=True)
        sims = util.cos_sim(query_emb, emb)[0]
        best = sims.argmax().item()
        self.activity_scores[best] += 0.5

class L3Memory:
    def __init__(self, encoder):
        self.entries = {}
        self.encoder = encoder

    def add(self, id, content):
        self.entries[id] = content

    def save_to_file(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)

    def load_from_file(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.entries = pickle.load(f)

class ACRMechanism:
    def __init__(self, l2, encoder, file_path="plik.txt", cache_path="plik_embeddings_cache.pkl"):
        self.l2 = l2
        self.encoder = encoder
        self.file_path = file_path
        self.cache_path = cache_path
        self.promoted = set()
        self._load_file_and_cache()

    def _load_file_and_cache(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.lines = [l.strip() for l in f if l.strip()]
        else:
            self.lines = []

        # cache embeddings
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("lines") == self.lines:
                self.embeddings = cache["embeddings"]
                return

        self.embeddings = (
            self.encoder.encode(self.lines, convert_to_tensor=True)
            if self.lines else None
        )
        with open(self.cache_path, "wb") as f:
            pickle.dump({"lines": self.lines, "embeddings": self.embeddings}, f)

    def inject_relevant_to_l2(self, iteration, query_emb):
        if not self.lines or self.embeddings is None or query_emb is None:
            # fallback dummy
            idx = self.l2.activity_scores.index(min(self.l2.activity_scores))
            elem = {"id": f"dummy_{iteration}", "content": f"Dummy {iteration}"}
            old = self.l2.slots[idx]
            if old:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(old["content"] + "\n")
            self.l2.add_to_slot(idx, elem, iteration)
            return

        sims = util.cos_sim(query_emb, self.embeddings)[0]
        for i in sims.argsort(descending=True)[:3]:
            if sims[i] < 0.2: break
            line = self.lines[i]
            lid = f"file_{hash(line)}"
            if lid in self.promoted: continue
            idx = self.l2.activity_scores.index(min(self.l2.activity_scores))
            old = self.l2.slots[idx]
            if old:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(old["content"] + "\n")
            self.l2.add_to_slot(idx, {"id": lid, "content": line}, iteration)
            self.promoted.add(lid)

# === Init models & memories ===

L3_FILE = "l3_memory.pkl"
L2_FILE = "plik.txt"
L2_CACHE = "plik_embeddings_cache.pkl"

print("Ładowanie modeli…")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2.eval()

l1 = L1Memory()
l2 = L2Memory(slot_count=5)
l3 = L3Memory(encoder)
acr = ACRMechanism(l2, encoder, L2_FILE, L2_CACHE)

l3.load_from_file(L3_FILE)
l1.add("id1","kot siedzi na dywanie")
l1.add("id2","pies biega po parku")
l2.add_to_slot(0, {"id":"slot0","content":"dziecko gra w piłkę"},0)
l3.add("kot","kot siedzi na dywanie i miauczy")
l3.add("pies","pies biega po parku z piłką")
l3.add("rower","człowiek jedzie na rowerze")

# === Flask app ===

app = Flask(__name__)

HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8"><title>AIram Chat</title></head>
<body>
<h2>AIram Chat</h2>
<div id="chat" style="height:300px;overflow:auto;border:1px solid #ccc;padding:5px"></div>
<input id="inp" style="width:80%"/>
<button onclick="s()">Wyślij</button>
<button onclick="showMemory()">Pokaż pamięć</button>
<div id="memory" style="white-space:pre-wrap; border:1px solid #999; margin-top:10px; padding:5px; max-height:300px; overflow:auto;"></div>

<script>
async function s(){
  let t=document.getElementById("inp"),c=document.getElementById("chat");
  c.innerHTML+='<div style="color:blue">Ty: '+t.value+'</div>';
  let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t.value})});
  let j=await r.json();
  c.innerHTML+='<div style="color:green">AIram: '+j.reply+'</div>';
  t.value='';c.scrollTop=c.scrollHeight;
}

async function showMemory(){
  let memDiv = document.getElementById("memory");
  memDiv.textContent = "Ładowanie pamięci...";
  let r = await fetch("/memory");
  let j = await r.json();
  memDiv.textContent = JSON.stringify(j, null, 2);
}
</script>
</body></html>
"""

@app.route("/")
def index():
    return HTML

def gen_resp(user_input):
    # Przygotuj prompt dialogowy z ostatnimi wiadomościami L1 + user input
    context = "\n".join(l1.get_all_contents()[-5:])  # ostatnie 5 wpisów z L1
    prompt = f"{context}\nUser: {user_input}\nAI:"

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    out = gpt2.generate(
        inputs, max_length=inputs.shape[1] + 50,  # pozwól wygenerować ~50 tokenów dalej
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    txt = tokenizer.decode(out[0], skip_special_tokens=True)

    # Wyciągnij wygenerowaną odpowiedź AI po "AI:"
    if "AI:" in txt:
        reply = txt.split("AI:")[-1].strip()
        # Jeśli dalej jest "User:" lub inne, odetnij to
        reply = reply.split("User:")[0].strip()
    else:
        reply = txt

    return reply

@app.route("/chat", methods=["POST"])
def chat():
    user = request.json.get("message","")
    it = len(l1.data) + 1
    l1.add(f"user_{it}", user)

    # Dopisanie user input do pliku
    with open(L2_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {user}\n")

    emb = encoder.encode(user, convert_to_tensor=True)
    acr.inject_relevant_to_l2(it, emb)
    l2.decay_and_consolidate_l2()
    l2.query_l3_and_activate_l2(emb, encoder)

    if user not in l3.entries.values():
        l3.add(f"user_{it}", user)
        l3.save_to_file(L3_FILE)

    reply = gen_resp(user)

    # Dopisanie AI reply do pliku
    with open(L2_FILE, "a", encoding="utf-8") as f:
        f.write(f"AI: {reply}\n")

    return jsonify({"reply": reply})
@app.route("/memory")
def memory():
    return jsonify({
        "L1": l1.data,
        "L2": l2.get_all_contents(),
        "L3": l3.entries
    })


if __name__ == "__main__":
    app.run(port=8081)
