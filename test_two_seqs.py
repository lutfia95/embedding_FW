from score_two_sequences import score_two_sequences

scores = score_two_sequences(
        "MKTAYIAK", "MKTAYIAQ",
        models=("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b", "esm1b-650m", "esm1v-650m-1", "esm1v-650m-2", "esm1v-650m-3", "esm1v-650m-4", "esm1v-650m-5", "esmc-300m", "esmc-600m", "esmc-6b"),
        device="auto",
)
print(scores)
print(scores["esm2-8m.mean.cosine_similarity"])