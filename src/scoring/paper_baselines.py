"""Paper equation adapters, explicitly not official full-system reproductions."""
import re


def answer_tokens(answer):
    return set(re.sub(r'[^\w\s]','',answer.lower()).split())


def ced(original,perturbed):
    """EVICT equations 2–3: Jaccard answer change, hard threshold <0.7."""
    a,b=answer_tokens(original),answer_tokens(perturbed)
    similarity=len(a&b)/len(a|b) if a|b else 1.
    return {'ced':float(similarity<.7)*(1-similarity),'ced_soft':1-similarity,'similarity':similarity}
