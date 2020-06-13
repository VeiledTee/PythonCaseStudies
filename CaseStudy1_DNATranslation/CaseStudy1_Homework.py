import string

# 1a
alphabet = " " + string.ascii_lowercase
# 1b
positions = dict(zip(alphabet, range(0, 27)))
# 1c
message = "hi my name is caesar"
"""
encodedMessage = ""
j = 0
for i in range(len(message)):
    for thing in alphabet:
        if message[i] == thing:
            for key in positions.keys():
                if key == thing:
                    j = positions[key] + 1
    encodedMessage += alphabet[j]
"""
# 1d
def encoding(toEncode, value):
    encodedMessage = ''
    j = 0
    for i in range(len(toEncode)):
        for thing in alphabet:
            if toEncode[i] == thing:
                for key in positions.keys():
                    if key == thing:
                        j = positions[key] + value
        encodedMessage += alphabet[j % 27]
    return encodedMessage
# 1e
print(message)
print(encoding(message, 3))
print(encoding(encoding(message, 3), -3))
