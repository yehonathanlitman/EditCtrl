# editctrl inference samples

Five `(video, mask)` pairs extracted from the Pexels/Videovo VPData val split,
downsampled to 49 frames at 720×480. The mask is the binary alpha-style mask
for the inpaint region (white = inpaint, black = keep). Use them directly
with any of the `*_editctrl.py` inference scripts:

```bash
python examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_editctrl.py \
    --input_video examples/wanvideo/model_inference/samples/sample_00_video.mp4 \
    --input_mask  examples/wanvideo/model_inference/samples/sample_00_mask.mp4 \
    --prompt "<prompt for sample 00, see below>" \
    --output_path sample_00_out.mp4
```

Checkpoints download automatically from Hugging Face on first run
(`thebluser/Wan*-editctrl`). Pass `--local_ckpt` / `--global_ckpt` to
override with a local file path.

## Wan 2.2 stability note

`Wan2.2-VACE-Fun-A14B_editctrl.py` defaults to **local-only** inference
(LoRA on `pipe.vace` and `pipe.vace2`). Excluding the global editctrl DiT
weights for Wan 2.2 proved more stable in our testing — the global path on
the MoE high/low experts occasionally introduced artifacts when combined
with the local LoRAs.

To opt back into the global path:

```bash
python examples/wanvideo/model_inference/Wan2.2-VACE-Fun-A14B_editctrl.py \
    --input_video samples/sample_00_video.mp4 \
    --input_mask  samples/sample_00_mask.mp4 \
    --prompt "..." \
    --enable_global
```

## Prompts

A machine-readable copy is in `prompts.json`.

### sample_00 — drone over a suburban landscape

> A drone captures a suburban landscape, showcasing rows of houses with grey roofs and white walls amidst green lawns and trees. The scene is set against a backdrop of majestic mountains with snow-capped peaks under an overcast sky, creating a tranquil atmosphere. As the drone moves, the landscape reveals a blend of residential and commercial buildings, with a network of roads connecting the community. The mountains stand as silent sentinels, their grandeur highlighted by the soft light filtering through the clouds, emphasizing the serene coexistence of human habitation and natural beauty.

### sample_01 — man eating a sunny-side-up egg

> A man is seated at a table, ready to eat a sunny-side-up fried egg garnished with dill, accompanied by neatly arranged cucumber slices and tomato wedges on a colorful plate. He is wearing a grey t-shirt with 'ADVENTURE' printed on it, indicating a casual dining experience. The setting includes a woven mat, adding to the ambiance. As time passes, the man, now in a grey t-shirt with 'ADVENTURE' and the number '127', continues to eat the egg, which is on a yellow plate with red and green patterns, suggesting a relaxed outdoor meal.

### sample_02 — barge with lumber on a calm river

> A large, weathered barge with a dark hull and red and white striped deck is seen cruising on a calm river, carrying a variety of brown wooden planks and beams. The barge, marked with the number '11', is equipped with a crane for loading or unloading materials. As it moves, the surrounding greenery and a partly cloudy sky create a serene atmosphere. The barge's deck is also lined with metal beams and wooden planks, indicating ongoing construction or maintenance work. The scene is peaceful, with the barge's solitary journey highlighted by the absence of other vessels or people.

### sample_03 — hand drawing on a digital tablet

> A hand with a black stylus pen is seen drawing on a digital tablet, which is connected to a keyboard with red backlit keys and a green power button, on a mouse pad featuring a cityscape design and a speedometer-like graphic. The setting appears to be a creative workspace, indicated by the presence of coins and a small white object on the wooden surface. The tablet's screen glows with a blue light, suggesting it is active. The scene remains consistent, focusing on the artistic process and the creative workspace environment.

### sample_04 — sunset over a pebble beach

> A serene sunset bathes a pebble-strewn beach in a golden glow, with the sun's reflection shimmering on the calm sea. A rugged rock formation stands in the foreground, silhouetted against the vibrant sky transitioning from blue to orange. Birds take flight, adding life to the tranquil scene. As time passes, the sun dips lower, casting a golden path across the water and sky, with the horizon ablaze in orange and yellow hues. The scene remains devoid of people and wildlife, emphasizing the natural beauty and peacefulness of the coastal landscape.
