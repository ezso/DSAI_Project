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
            "ch": "(?:ch|h|c|n)",
            "ck": "(?:ck|k)",
            "ſt": "(?:ſt|st|f)",
            "ni": "(?:ni|m)",
            "li": "(?:li|h)",
            "tt": "(?:tt|n)",
        }
    
    def build_ocr_tolerant_pattern(self, term: str) -> str:
        """
        Build an OCR-tolerant regex pattern for a given stem.

        Features:
        - Stem can appear anywhere in a word
        - Handles optional wrappers like parentheses, brackets, braces, angled symbols, and top/bottom quotes
        - Allows up to 3 consecutive OCR/noise characters between letters
        - Handles OCR confusions (single and multi-character)
        """
        term = str(term)  # ensure string
        lower = term.lower()

        # sort multi_confusion keys by length desc so longer matches win (e.g. 'rn' before 'r')
        multi_keys = sorted(self.multi_confusions.keys(), key=len, reverse=True)

        parts = []
        i = 0
        while i < len(term):
            matched_multi = False
            for k in multi_keys:
                if lower.startswith(k, i):
                    # insert the regex fragment (it's already a regex)
                    parts.append(self.multi_confusions[k])
                    i += len(k)
                    matched_multi = True
                    break
            if matched_multi:
                continue

            ch = term[i]
            lc = ch.lower()
            if lc in self.single_confusions:
                parts.append(self.single_confusions[lc])
            else:
                parts.append(re.escape(ch))
            i += 1

        # noisy characters allowed between the pattern parts (0..3)
        noisy = r'[\s\-\¬\�\?\«\>\<!:/|*#@~]{0,3}'
        core = noisy.join(parts)

        # allow some letters before/after the stem (stem can be anywhere in a word)
        word_prefix = r'\w{0,15}'
        word_suffix = r'\w{0,15}'

        # small wrappers/noise before/after (max 3)
        opening_wrappers = r'[\(\[\{<"\'«‹„“”‘’\s\-\¬\�\?\>!:/|*#@~]{0,3}'
        closing_wrappers = r'[\)\]\}>\”’»“‘\s\-\¬\�\?\<!:/|*#@~]{0,3}'

        final_pattern = f"{opening_wrappers}{word_prefix}{core}{word_suffix}{closing_wrappers}"
        return final_pattern
    
    def generate_response(self, ocr_text):
        results = {"success": False}
        for idx, term in enumerate(self.terms, start=1):
            pat = re.compile(self.build_ocr_tolerant_pattern(term), re.IGNORECASE)
            matches = sorted(set(pat.findall(ocr_text)))
            results[f"word{idx}"] = matches
        if any(results[f"word{idx}"] for idx in range(1, len(self.terms) + 1)):
            results["success"] = True
        return results
