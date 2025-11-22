import hashlib


def extract_features(input):
    words = input.split()
    features = [word.lower() for word in words]
    return features


def simhash(input):
    features = extract_features(input)

    hashes = [hashlib.sha1(feature.encode("utf-8")).hexdigest() for feature in features]

    concatenated_hash = "".join(hashes)
    simhash = hashlib.sha1(concatenated_hash.encode("utf-8")).hexdigest()

    return simhash


def compare_simhashes(simhash1, simhash2):
    int_simhash1 = int(simhash1, 16)
    int_simhash2 = int(simhash2, 16)

    print(bin(int_simhash1 ^ int_simhash2))

    distance = bin(int_simhash1 ^ int_simhash2).count("1")

    return distance


text1 = "The quick brown fox jumps over the lazy dog."
text2 = "The quick brown fox jumps over the fast cat."
simhash1 = simhash(text1)
simhash2 = simhash(text2)

distance = compare_simhashes(simhash1, simhash2)
print(f"Distance between simhashes: {distance}")
