
def make_chunks(sentences, max_char = 300):
    chunks = []
    current = ""
    for s in sentences:
        if len(current)+ len(s)> max_char and current:
         chunks.append(current)
         current = ''
        current += s
    if current:
       chunks.append(current)
    return chunks

def split_long(s, max_char=300):
    pieces=[]
    for i in range(0, len(s), max_char):
        pieces.append(s[i:max_char +i])
    return pieces