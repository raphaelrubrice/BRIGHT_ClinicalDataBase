import re

r = re.compile(r'(?:^|[\-:–—|/]\s*)(?P<label>initial|terminal)\b|(?:^|[\-:–—|/\s]\s*)(?P<label2>P\d+)\b', re.I | re.M)
m = r.search('Le patient est adressé pour bilan initial.')
print(f"Match: {m}")
if m:
    print(f"Group: {m.group()}")
    print(f"Start: {m.start()}, End: {m.end()}")
