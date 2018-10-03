import re

def natsort(strings): # natural sort
    ttime = re.compile(r"_t(\d+)\.")

    def _natkey(term):
        res = ttime.search(term)
        return int(res.group(1)) if res is not None else term

    return sorted(strings, key=_natkey)
