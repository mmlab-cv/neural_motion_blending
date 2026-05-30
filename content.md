## 🦴 (A) In-Skeleton Blending

Same skeleton, smoother transitions. We blend the latent representations of two motions defined on an **identical** skeleton, producing seamless temporal transitions that stay faithful to both ends. Put against *AnyTop* and *Blender NLA*, our method eliminates rotation artifacts and better respects the boundary poses — the difference is visible frame by frame.

[MEDIA:inSkel1,inSkel2,inSkel3]{**Examples 1–3.** From left to right: Triceratops (Walking → Running), King Cobra (Circle → Bite), Dragon (Attack → Attack). Each clip shows the reference motion, our blend, and the two baselines side by side. Notice how *AnyTop* drifts from both endpoints and *Blender* introduces joint-level artifacts at the transition — our method stays clean throughout.}

[SPACING:large]

## 🐊 (B) Cross-Skeleton Blending

This is where things get interesting. We blend motions across characters with **completely different** skeletal topologies — no shared rig, no manual correspondence. The model learns a shared latent space across heterogeneous characters and interpolates directly in it, preserving timing, rhythm, and style even when the source and target don't share a single bone.

[MEDIA:xSkel1,xSkel2,xSkel3]{**Examples 1–3.** Crab → Polar Bear, Leopard → Puppy, Raptor → Skunk. Watch how each character progressively acquires the locomotion style of the target: gait phase synchronizes, limb coordination shifts, and the motion reads naturally on both topologies throughout the blend.}

[SPACING:medium]

[MEDIA:xSkelBonus]{🦀 **Bonus — Crab (Attack) → Raptor (Roar).** This crab means business. Don't mess with it.}

[SPACING:large]

## 🎯 (C) Retargeting

Retargeting wasn't the primary training objective — yet the learned representation handles it out of the box. We pit our method against *Motion2Motion*, a dedicated retargeting baseline, and the results speak for themselves: our model preserves the **characteristic style** of the source motion while adapting it to the target topology, something *Motion2Motion* consistently struggles with.

[MEDIA:retarget1,retarget2]{**Comparison with Motion2Motion — Examples 1–2.** Bird → Bat (fly loop) and Ostrich → Flamingo (flap wings, one leg bent). *Motion2Motion* stays anchored to the source domain and misses key stylistic traits of the target. Our method picks them up automatically: wing tempo, body posture, and even the flamingo's characteristic bent leg all transfer correctly.}

[SPACING:medium]

### 🌍 One Motion, Many Characters

One source motion, retargeted to a whole cast of different topologies at once. Style and timing stay consistent across wildly different body structures — no per-character tuning required.

[MEDIA:retargetMany]

[SPACING:large]

## 🚀 (Extra) Zero-Shot Transfer from Unseen Topologies

No human examples seen during training — yet the model transfers Mixamo motions to a diverse set of Truebones animal characters with entirely different skeletal structures. This demonstrates that the learned representation **generalizes beyond the observed topology distribution**, opening the door to plug-and-play animation transfer for characters the model has never encountered.

[MEDIA:zeroShot]
