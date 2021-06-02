def nbeditops(s1, s2):
    import Levenshtein as L
    d = 0
    i = 0
    s = 0
    for op in L.editops(s1, s2):
        if op[0] == 'delete':
            d += 1
        elif op[0] == 'insert':
            i += 1
        elif op[0] == 'replace':
            s += 1
    return d, i, s


def cer(hyps, refs):
    delete = 0
    insert = 0
    substitute = 0
    nbchar = 0
    for h, r in zip(hyps, refs):
        d, i, s = nbeditops(r, h)
        delete += d
        insert += i
        substitute += s
        nbchar += len(r)
    cer = (delete + insert + substitute) / nbchar
    correct = nbchar - substitute - delete
    return {'CER': cer, 'Cor': correct, 'Sub': substitute, 'Ins': insert,
            'Del': delete}


def wer(hyps, refs):
    delete = 0
    insert = 0
    substitute = 0
    correct = 0
    nbwords = 0
    for h, r in zip(hyps, refs):
        results = wer_sent(r, h)
        delete += results['Del']
        insert += results['Ins']
        substitute += results['Sub']
        correct += results['Cor']
        nbwords += len(r.split())
    wer = (delete + insert + substitute) / nbwords
    return {'WER': wer, 'Cor': correct, 'Sub': substitute, 'Ins': insert,
            'Del': delete}


def wer_sent(ref, hyp, debug=False):
    '''
    Computes the word error rate between reference and hypothesis
    Modified from SpacePineapple
    (https://progfruits.blogspot.com/2014/02/word-error-rate-wer-and-word.html)
    '''
    SUB_PENALTY = 100
    INS_PENALTY = 75
    DEL_PENALTY = 75

    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY
                insertionCost = costs[i][j-1] + INS_PENALTY
                deletionCost = costs[i-1][j] + DEL_PENALTY

                costs[i][j] = min(substitutionCost, insertionCost,
                                  deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    wer_result = round((numSub + numDel + numIns) / (float)(len(r)), 3)
    return {'WER': wer_result, 'Cor': numCor, 'Sub': numSub, 'Ins': numIns,
            'Del': numDel}
