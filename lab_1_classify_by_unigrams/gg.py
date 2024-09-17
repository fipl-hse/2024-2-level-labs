tokens = ['a', 'f', 'g', 'a', 'u']
d = {}
all_tokens = len(tokens)
for token in tokens:
    if token in d.keys():
        d[token] = 1 + d[token]
    else:
        d[token] = 1
for token in d:
    d[token] /= d[all_tokens]

