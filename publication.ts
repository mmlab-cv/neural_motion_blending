import type { Publication, Theme } from "@/_internal/types";
import { COMING_SOON } from "./md-paper/types";

const theme: Theme = {
  accentColor:     "#0a4b7c",
  pageBackground:  "#ffffff",
  blockBackground: "#f7f7f7",
  baseFontSize:    16,
  titleFontSize:   48,
  authorFontSize:  18,
  headingFontSize: 22,
  abstractFontSize:16,
  contentFontSize: 16,
  mediaTitleFontSize:  16,
  mediaCaptionFontSize:14,
  contentMaxWidth: 1200,
  bodyFont:        "Lato, sans-serif",
  headingFont:     '"Patua One", serif',
};

const publication: Publication = {
  title: "Neural Motion Blending Across Arbitrary Character Topologies",
  theme,

  authors: [
    ["L. Cazzola",    "https://scholar.google.com/citations?user=fsnsqoYAAAAJ&hl=en", "1"],
    ["G. Martinelli", "https://scholar.google.com/citations?user=WG3OkQ4AAAAJ&hl=en", "1,2"],
    ["N. Conci",      "https://scholar.google.com/citations?user=mR1GK28AAAAJ&hl=en", "1,2"],
  ],

  affiliations: [
    ["1", "University of Trento"],
    ["2", "CNIT"],
  ],

  venue: "CGI • Computer Graphics International",
  year:  "2026",

  paper:         "/media/Luca_Cazzola-Neural_Motion_Blending-Paper.pdf",
  pdf:           undefined,
  code:          "https://github.com/mmlab-cv/neural_motion_blending",
  supplementary: "/media/Luca_Cazzola-Neural_Motion_Blending-Supplementary.zip",

  siteUrl: "https://github.com/mmlab-cv",
  siteLabel: "← Check out other MMLab works!",

  teaserIndex: 1,

  abstract: "Motion blending in character animation enables the synthesis of new motions by interpolating between existing examples. Current methods are typically restricted to fixed skeleton topologies, requiring identical or near-identical skeletal structures across characters. We present a novel framework for motion blending across heterogeneous skeletons. The proposed architecture combines a semantic encoder, which extracts a compact per-frame latent representation of the global motion state, with a diffusion-based decoder, which reconstructs character-specific motion conditioned on this latent code. At inference time, blended motions are obtained by interpolating the latent representations of two input motions. We train and evaluate the method on the Truebones Zoo dataset using motions defined on both same and distinct skeleton topologies, demonstrating the ability to achieve smooth and plausible blending in a variety of different scenarios.",

  media: [
    // 1 — teaser
    {
      type:  "video",
      src:   "/media/teaser.mp4",
      id:    "teaser",
      title: "Teaser",
      caption: "Our approach enables, for the first time, motion blending across arbitrary character topologies.",
    },

    // (A) In-Skeleton Blending — 2, 3, 4
    {
      type:  "video",
      src:   "/media/inSkel_comparison1.mp4",
      id:    "inSkel1",
      title: "In-Skeleton Example 1",
      caption: "Triceratops (ref: Walking → tgt: Running). AnyTop struggles to maintain fidelity to the reference and target motions. Blender produces a rotation artifact on the front left leg at the transition. Our method, by contrast, is faithful to both inputs and yields a cleaner blending region.",
    },
    {
      type:  "video",
      src:   "/media/inSkel_comparison2.mp4",
      id:    "inSkel2",
      title: "In-Skeleton Example 2",
      caption: "King Cobra (ref: Circle → tgt: Bite). AnyTop struggles to maintain fidelity to the reference and target motions. Blender fails to preserve reasonable root joint rotation at the transition. Our method is superior on both counts.",
    },
    {
      type:  "video",
      src:   "/media/inSkel_comparison3.mp4",
      id:    "inSkel3",
      title: "In-Skeleton Example 3",
      caption: "Dragon (ref: Attack → tgt: Attack). AnyTop struggles to maintain fidelity to the reference and target motions. Blender introduces a rotation error in the right wing. Our method is superior on both counts.",
    },

    // (B) Cross-Skeleton Blending — 5, 6, 7, 8
    {
      type:  "video",
      src:   "/media/xSkel_comparison1.mp4",
      id:    "xSkel1",
      title: "Cross-Skeleton Example 1",
      caption: "ref: Crab (Stride) → tgt: Polar Bear (Run). The crab begins with its normal striding motion; upon transition it acquires meaningful quadrupedal coordination while maintaining temporal alignment with the polar bear's motion and remaining faithful to the crab's topology.",
    },
    {
      type:  "video",
      src:   "/media/xSkel_comparison2.mp4",
      id:    "xSkel2",
      title: "Cross-Skeleton Example 2",
      caption: "ref: Leopard (Walking) → tgt: Puppy (Running). The leopard begins with a walking motion; upon transition, it adapts to a running gait while maintaining temporal coherence and semantic alignment with the puppy's running style.",
    },
    {
      type:  "video",
      src:   "/media/xSkel_comparison3.mp4",
      id:    "xSkel3",
      title: "Cross-Skeleton Example 3",
      caption: "ref: Raptor (Slow Walking) → tgt: Skunk (Running). The raptor begins with a slow walking motion; upon transition, it adapts to a quadrupedal running gait. Notice how the raptor's legs synchronize with the skunk's hind legs.",
    },
    {
      type:  "video",
      src:   "/media/xSkel_extra.mp4",
      id:    "xSkelBonus",
      title: "Cross-Skeleton Bonus",
      caption: "ref: Crab (Attack) → tgt: Raptor (Roar). This crab means business. Don't mess with it!",
    },

    // (C) Retargeting — 9, 10, 11
    {
      type:  "video",
      src:   "/media/transfer_comparison2.mp4",
      id:    "retarget1",
      title: "Retargeting Example 1",
      caption: "ref: Bird → tgt: Bat (Fly loop). Motion2Motion produces a flying motion that stays within the bird's domain and struggles to adapt to the bat's faster tempo. Our method automatically adjusts the bird's posture to a more bat-like stance and synchronizes wing movement.",
    },
    {
      type:  "video",
      src:   "/media/transfer_comparison3.mp4",
      id:    "retarget2",
      title: "Retargeting Example 2",
      caption: "ref: Ostrich → tgt: Flamingo (Flap wings, one leg bent). Motion2Motion only partially reproduces the 'one leg bent' trait, which appears to be frozen in a walking pose mid-air. Our method reproduces both the wing-flapping motion and correctly bends the right leg in a flamingo-like manner.",
    },
    {
      type:  "video",
      src:   "/media/transfer_comparison_many.mp4",
      id:    "retargetMany",
      title: "One Motion, Many Characters",
      caption: "A single source motion retargeted to a diverse set of character topologies, demonstrating consistency of style and timing across large structural variation.",
    },

    // (Extra) Zero-Shot Transfer — 12
    {
      type:  "video",
      src:   "/media/human2animal.mp4",
      id:    "zeroShot",
      title: "Zero-Shot Transfer from Unseen Topologies",
      caption: "Without any human examples during training, the model transfers Mixamo motions to multiple Truebones characters with different skeletal structures, demonstrating extrapolation beyond the observed topology distribution.",
    },
  ],
};

export default publication;
