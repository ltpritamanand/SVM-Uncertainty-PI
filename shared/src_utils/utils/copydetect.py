from copydetect import CodeFingerprint, compare_files, utils

# Create fingerprints for the file itself
fp1 = CodeFingerprint("copydetect.py", noise_threshold=25, guarantee_threshold=1)
fp2 = CodeFingerprint("copydetect.py", noise_threshold=25, guarantee_threshold=1)

# Compare the fingerprints
token_overlap, similarities, slices = compare_files(fp1, fp2)

# Display the results
print(f"Token overlap: {token_overlap}")
print(f"Similarity scores: {similarities}")

# Highlight and display the overlapping code sections
code1_highlighted, _ = utils.highlight_overlap(fp1.raw_code, slices[0], ">>", "<<")
code2_highlighted, _ = utils.highlight_overlap(fp2.raw_code, slices[1], ">>", "<<")

print("Highlighted similarities in copydetect.py:")
print(code1_highlighted)
print("\nHighlighted similarities in copydetect.py:")
print(code2_highlighted)
