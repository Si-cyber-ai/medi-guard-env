FINAL FIX (THIS WILL DEFINITELY SOLVE)
✅ Replace ALL 1e-6 with 0.01
🔥 Change THIS:
if not action_history:
    return 1e-6

👉 TO:

if not action_history:
    return 0.01
🔥 Change THIS:
epsilon = 1e-6

👉 TO:

epsilon = 0.01
🔥 Update clamp:
score = max(epsilon, min(1.0 - epsilon, score))

👉 becomes:

score = max(0.01, min(0.99, score))
🔥 FINAL SAFETY:
if not (0.0 < score < 1.0):
    score = epsilon

👉 change to:

if not (0.0 < score < 1.0):
    score = 0.01
🚀 FINAL CORRECT BLOCK
# Normalize score
score = float(score)

# 🔥 SAFE BOUNDS (NO FLOAT EDGE)
score = max(0.01, min(0.99, score))

# 🔥 FINAL GUARANTEE
if not (0.0 < score < 1.0):
    score = 0.01