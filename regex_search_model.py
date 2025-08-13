import re
import json

class RegexSearchModel:
    def __init__(self, terms):
        self.terms = terms
        self.confusions = {
            "a": "[aä]", 
            "ä": "[äa]", 
            "b": "[b]", 
            "c": "[cce]", 
            "d": "[do]", 
            "e": "[eéc]", 
            "f": "[fſ]", 
            "g": "[gq]", 
            "h": "[hn]", 
            "i": "[i1ltn]", 
            "j": "[j]", 
            "k": "[kck]", 
            "l": "[lt1i]", 
            "m": "[mnnrn]", 
            "n": "[nhu]", 
            "o": "[o0]", 
            "p": "[p]", 
            "q": "[qg]", 
            "r": "[r]", 
            "s": "[sſf]", 
            "ß": "[ßssſ]", 
            "t": "[t1i]", 
            "u": "[uüvn]", 
            "ü": "[üu]", 
            "v": "[vu]", 
            "w": "[wvv]", 
            "x": "[x]", 
            "y": "[y]", 
            "z": "[z2]",
        }
    
    def build_ocr_tolerant_pattern(self, term):
        base = term.lower()
        pattern = ""
        for ch in base:
            if ch in self.confusions:
                pattern += self.confusions[ch]
            else:
                pattern += ch
        # Allow any suffix (cognates, compounds) and minor internal noise
        pattern = pattern.replace("", ".{0,1}")  # small tolerance between chars
        return r"\b" + pattern + r"\w*\b"
    
    def generate_response(self, ocr_text):
        results = {"success": False}
        for idx, term in enumerate(self.terms, start=1):
            pat = re.compile(self.build_ocr_tolerant_pattern(term), re.IGNORECASE)
            matches = sorted(set(pat.findall(ocr_text)))
            results[f"word{idx}"] = matches
        if any(results[f"word{idx}"] for idx in range(1, len(self.terms) + 1)):
            results["success"] = True
        return results
