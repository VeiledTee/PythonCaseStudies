def translate(sequence):
    """
    Translate a string containing a nucleotide sequence into a string containing
    the corresponding amino acids. Nucleotides are translated in triplets using
    the table dictionary; each amino acid 4 is encoded with a string length of 1

    :param sequence: A sequence of nucleotides
    :return: string containing corresponding amino acids
    """
    table = {'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
             'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
             'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
             'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
             'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
             'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
             'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
             'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
             'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
             'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
             'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
             'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
             'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
             'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
             'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
             'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W', }

    protein = ""
    if len(sequence) % 3 == 0:
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i + 3]
            protein += table[codon]
    else:
        print("Your input is not divisible by 3. Try a different input.")
    return protein

def readSequence(fileInput):
    """
    Reads and returns the input sequence with special characters removed
    :param fileInput: input sequence
    :return: input sequence with special characters removed
    """
    with open(fileInput, "r") as t:
        sequence = t.read()
    sequence = sequence.replace('\n', "")
    sequence = sequence.replace('\r', "")
    return sequence

prt = readSequence("protein.txt")
dna = readSequence("dna.txt")
print(translate(dna))  # returns ''
# checking on NCBI website, translation starts at nucleotide 21 and ends at 938
# so we have to slice from nucleotide 20 to 937 (cuz indexing)
# print(translate(dna[20:938])) returns a stop codon:
# the functions of stop codons is to tell someone readin the sequence that this is where to stop
# need to stop reading this sequence before the stop codon
print(prt == translate(dna[20:935]))  # end 3 before 938, yields True
# alt approach:
# omit the last character of the sequence
print(prt == translate(dna[20:938])[:-1])  # True
print(translate(dna[20:935]) == translate(dna[20:938])[:-1])
