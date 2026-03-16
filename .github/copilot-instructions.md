## Colour assignments — this project (seq2seq architecture)

When drawing diagrams for this project, apply this domain-to-colour
mapping consistently so all figures share the same visual language.
These assignments follow the global Cisco brand guidelines in
~/copilot-instructions.md.

Domain colours (fill at opacity 35 for boxes, opacity 20 for zone backgrounds):

  Encoder domain (BiLSTM, embeddings):   Cisco Blue      stroke: #02C8FF   fill: #02C8FF @ op 35
  Decoder domain (LSTM, output head):    Medium Blue     stroke: #0A60FF   fill: #0A60FF @ op 35
  Attention mechanism boxes:             Medium Blue     stroke: #0A60FF   fill: #0A60FF @ op 35
  Bridge / projection components:        30% Midnight    stroke: #6B6B6B   fill: #B4B9C0 @ op 35
  Final output / logits box:             Cisco Red       stroke: #EB4651   fill: #EB4651 @ op 40  strokeWidth: 2
  Neutral / shared components:           White           stroke: #6B6B6B   fill: #FFFFFF @ op 100
  Input / output token boxes:            White           stroke: #6B6B6B   fill: #FFFFFF @ op 100
  Concat / operation nodes:              White           stroke: #6B6B6B   fill: #FFFFFF @ op 100
  Recurrent state side boxes:            Light grey      stroke: #6B6B6B   fill: #F0F0F0 @ op 100
  Note / annotation boxes:              Light grey      stroke: #6B6B6B   fill: #F0F0F0 @ op 100

Zone background rectangles (lowest z-order, drawn first):

  Encoder zone:   fill: #02C8FF @ op 20   stroke: #02C8FF   strokeWidth: 1
  Decoder zone:   fill: #0A60FF @ op 20   stroke: #0A60FF   strokeWidth: 1
  Attention zone: fill: #0A60FF @ op 20   stroke: #0A60FF   strokeWidth: 1

Zone label text colour (must match zone stroke, not black):

  Encoder zone label:   strokeColor: #02C8FF
  Decoder zone label:   strokeColor: #0A60FF
  Attention zone label: strokeColor: #0A60FF

All body text and arrows: strokeColor: #07182D (Midnight Blue)
All text on any coloured fill: strokeColor: #07182D (never use a coloured strokeColor on body text)
fontFamily: 2 on every text element (geometric sans-serif — closest to CiscoSansTT)
roughness: 0 on every element including standalone text
