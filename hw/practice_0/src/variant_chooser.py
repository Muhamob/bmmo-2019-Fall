# Dmitry Vetrov -> 3 variant
# Thomas Bayes -> 1 variant

abc = 'abcdefghijklmnopqrstuvwxyz'
name = input('What is your name?\n')
letter_indices = [abc.index(l) for l in name.lower() if l in abc]
variant = 1 + sum(letter_indices) % 3
print('It\'s %d variant' % variant)
