# Go1 Challenge - Isaac Sim environment
Creation of the assets for the arena environment here. 

Notes on how the env is created:

Generation of the world: `arena_5x5.usd`
File to generate: `create_arena_usd.py`

The arena uses 60 unique tags: 15 on each wall. IDs 0--13 come from
`assets/april_tags.usd`; IDs 14--59 are generated with the same tag36h11 MDL
material. The 20 cm tag prints have a 12.5 cm edge-to-edge gap.

**Assets Needed**
*assets/april_tags.usd*

