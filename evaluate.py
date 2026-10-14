import csv
import hashlib
import json
import os
import sys
import time
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

from rag_answer import vectorstore, llm, prompt, as_text

TEST_FILE = "eval/test_set.csv"
CACHE_FILE = "eval/answer_cache.json"
REFUSAL = "i could not find this in the document"

known = {"--retrieval-only"}
bad = [a for a in sys.argv[1:] if a.startswith("-") and a not in known]
if bad:
    raise SystemExit(f"Unknown option {bad}. Example: python evaluate.py 5 --retrieval-only")
args = [a for a in sys.argv[1:] if not a.startswith("-")]
K = int(args[0]) if args else 5
RETRIEVAL_ONLY = "--retrieval-only" in sys.argv
suffix = "_retrieval" if RETRIEVAL_ONLY else ""
OUT_FILE = f"eval/results_k{K}{suffix}.csv"

cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)


class QuotaStop(Exception):
    pass


def call_llm(prompt_value):
    """Return (answer, was_cached). Saved answers are reused, so reruns cost nothing."""
    key = hashlib.sha256(prompt_value.to_string().encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key], True
    for _ in range(3):
        try:
            answer = as_text(llm.invoke(prompt_value).content)
            cache[key] = answer
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            return answer, False
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:
                raise QuotaStop()
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("  (per-minute limit hit, waiting 60s)")
                time.sleep(60)
                continue
            raise
    raise QuotaStop()


def split_list(text):
    return [x.strip() for x in (text or "").split(";") if x.strip()]


def rate(num, den):
    return f"{num}/{den}" + (f" = {100 * num / den:.0f}%" if den else "")


with open(TEST_FILE, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

results = []
stopped = False

for r in rows:
    question = r["question"]
    answerable = r["answerable"].strip().lower() == "yes"
    exp_pages = split_list(r["expected_page"])
    keywords = [w.lower() for w in split_list(r["expected_keywords"])]

    docs = vectorstore.similarity_search(question, k=K)
    pages = [str(d.metadata["page"]) for d in docs]
    unique_pages = list(dict.fromkeys(pages))

    rank = None
    if answerable and exp_pages:
        for i, p in enumerate(pages, start=1):
            if p in exp_pages:
                rank = i
                break
        retrieval_hit = "yes" if rank else "no"
    else:
        retrieval_hit = "n/a"

    answer, refused, keyword_hit = "", False, "n/a"

    if RETRIEVAL_ONLY:
        diagnosis = {
            "yes": "pass",
            "no": "FAIL: retrieval (right page not retrieved)",
        }.get(retrieval_hit, "n/a")
    else:
        context = "\n\n".join(f"[Page {d.metadata['page']}] {d.page_content}" for d in docs)
        try:
            answer, cached = call_llm(prompt.invoke({"context": context, "question": question}))
        except QuotaStop:
            stopped = True
            break
        refused = REFUSAL in answer.lower()
        if answerable and keywords:
            keyword_hit = "yes" if all(w in answer.lower() for w in keywords) else "no"

        if not answerable:
            diagnosis = "pass" if refused else "FAIL: should have refused"
        elif retrieval_hit == "no":
            diagnosis = "FAIL: retrieval (right page not retrieved)"
        elif refused:
            diagnosis = "FAIL: refused although answerable"
        elif keyword_hit == "no":
            diagnosis = "CHECK: keywords missing (generation?)"
        elif retrieval_hit == "n/a" and keyword_hit == "n/a":
            diagnosis = "UNCHECKED: add expected page/keywords"
        else:
            diagnosis = "pass"
        if not cached:
            time.sleep(4)  # stay under the per-minute limit

    results.append({
        "id": r["id"], "category": r["category"], "question": question,
        "answerable": "yes" if answerable else "no",
        "expected_page": r["expected_page"],
        "retrieved_pages": ", ".join(unique_pages),
        "rank": rank if rank else "",
        "retrieval_hit": retrieval_hit, "keyword_hit": keyword_hit,
        "refused": "yes" if refused else "no",
        "diagnosis": diagnosis, "answer": answer, "human_grade": "",
    })
    print(f"{r['id']:>3} [{r['category']}] {diagnosis}")

if stopped:
    print(f"\nDaily quota reached after {len(results)} of {len(rows)} questions.")
    print("Rerun after the quota resets. Finished answers are cached, so only the rest will cost requests.")

if not results:
    sys.exit()

with open(OUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

ret = [x for x in results if x["retrieval_hit"] != "n/a"]
hits = [x for x in ret if x["retrieval_hit"] == "yes"]
print("\n" + "=" * 60)
print(f"SUMMARY (top-k = {K}, {'retrieval only' if RETRIEVAL_ONLY else 'full pipeline'})")
print("Retrieval hit rate :", rate(len(hits), len(ret)))
if hits:
    print(f"Average rank of hit: {sum(int(x['rank']) for x in hits) / len(hits):.1f}")
if not RETRIEVAL_ONLY:
    kw = [x for x in results if x["keyword_hit"] != "n/a"]
    unans = [x for x in results if x["answerable"] == "no"]
    print("Keyword match rate :", rate(sum(x["keyword_hit"] == "yes" for x in kw), len(kw)))
    print("Refusal accuracy   :", rate(sum(x["refused"] == "yes" for x in unans), len(unans)))
print("\nBy diagnosis:")
for label, n in Counter(x["diagnosis"] for x in results).most_common():
    print(f"  {n:>2}  {label}")
print(f"\nDetails saved to {OUT_FILE}")