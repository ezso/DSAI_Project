import re

class RegexSearchModel:
    def __init__(self, terms):
        self.terms = terms
        self.single_confusions = {
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
        
        self.multi_confusions = {
            "rn": "(?:rn|m)",
            "vv": "(?:vv|w)",
            "ri": "(?:ri|n)",
            "cl": "(?:cl|d)",
            "ch": "(?:ch|h)",
            "ck": "(?:ck|k)",
            "ſt": "(?:ſt|st|f)",
            "ni": "(?:ni|m)",
            "li": "(?:li|h)",
            "tt": "(?:tt|n)",
        }
    
    def build_ocr_tolerant_pattern(self, term: str) -> str:
        base = term.lower()
        pattern = ""

        i = 0
        while i < len(base):
            # first check multi-character confusions
            matched = False
            for multi, replacement in self.multi_confusions.items():
                if base.startswith(multi, i):
                    pattern += replacement
                    i += len(multi)
                    matched = True
                    break

            if matched:
                continue

            # then check single-character confusions
            ch = base[i]
            if ch in self.single_confusions:
                pattern += self.single_confusions[ch]
            else:
                pattern += ch
            i += 1

        # allow small OCR noise between characters
        pattern = ".{0,1}".join(pattern)

        # word boundaries, but Unicode-safe
        return r"(?<!\w)" + pattern + r"\w*(?!\w)"
    
    def generate_response(self, ocr_text):
        results = {"success": False}
        for idx, term in enumerate(self.terms, start=1):
            pat = re.compile(self.build_ocr_tolerant_pattern(term), re.IGNORECASE)
            matches = sorted(set(pat.findall(ocr_text)))
            results[f"word{idx}"] = matches
        if any(results[f"word{idx}"] for idx in range(1, len(self.terms) + 1)):
            results["success"] = True
        return results
