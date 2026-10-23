"""Eight absolute, ternary genes for the fixed-topology hand family."""
import copy
import hashlib
import json
import math
import random

NAMES = ("palm_width", "finger_spread", "finger_length", "proximal_fraction",
         "thumb_length", "thumb_lateral", "thumb_axis", "pad_width")
BASELINE = (0,) * 8


def validate(code):
    code = tuple(code)
    if len(code) != 8 or any(type(x) is not int or x not in (-1, 0, 1) for x in code):
        raise ValueError("Expected eight integer levels in {-1, 0, 1}")
    return code


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def mutate(code, rng=None):
    """One adjacent level per birth; never jump from low directly to high."""
    rng = rng or random.Random()
    child = list(validate(code)); index = rng.randrange(8)
    child[index] = rng.choice([x for x in (-1, 0, 1) if abs(x-child[index]) == 1])
    return tuple(child)


def decode(baseline, code, palm_width_m):
    """Always decode from the immutable baseline, never from a parent phenotype.

    This family's hand plane is X-Z, transverse width is Z, and Y is normal.
    The proximal phalanx is link_d_0. Total digit length includes its welded tip.
    Segment lengths mean joint-to-joint spans, with the terminal shaft as the last
    span. Bone radii stay fixed; child attachment offsets follow the new spans.
    """
    code = validate(code)
    if not math.isfinite(palm_width_m) or palm_width_m <= 0:
        raise ValueError("A positive measured baseline palm width is required")
    hand = copy.deepcopy(baseline)
    links = {x['name_code']: x for x in hand['links']}
    expected = {f'link_{d}_{s}' for d in range(1, 6) for s in range(4)}
    if set(links) != expected:
        raise ValueError("This schema requires the calibrated five-digit, four-segment genotype")
    roots = [links[f'link_{d}_0'] for d in range(1, 6)]
    center = (min(x['joint_origin_translation'][2] for x in roots) +
              max(x['joint_origin_translation'][2] for x in roots)) / 2
    if code[0]:
        for root in roots:
            z = root['joint_origin_translation'][2]
            root['joint_origin_translation'][2] = center + (z-center)*(1+.1*code[0])
    links['link_1_0']['joint_origin_translation'][2] += .05*palm_width_m*code[5]
    # Fan opening relative to the middle finger; a shared rotation would only
    # rotate the hand, not change spread. Middle-finger root stays unchanged.
    for d, sign in ((2, 1), (3, 0), (4, -1), (5, -1)):
        links[f'link_{d}_0']['joint_origin_rpy'][1] += sign*math.radians(5)*code[1]
    for d in range(1, 6):
        chain = [links[f'link_{d}_{s}'] for s in range(4)]
        old = [math.sqrt(sum(v*v for v in c['joint_origin_translation'])) for c in chain[1:]]
        old.append(chain[-1]['geometry_length'])
        scale = 1+.1*code[4 if d == 1 else 2]
        new = [v*scale for v in old]
        if d != 1 and code[3]:
            total = sum(new); fraction = new[0]/total + .05*code[3]
            if not 0 < fraction < 1:
                raise ValueError("Invalid proximal fraction")
            rest = sum(new[1:]); new = [total*fraction] + [v*total*(1-fraction)/rest for v in new[1:]]
        for i, part in enumerate(chain):
            if new[i] != old[i]:
                part['geometry_length'] += new[i]-old[i]
                if part['geometry_length'] <= 0:
                    raise ValueError("Nonpositive shaft length")
                if i < 3:
                    child = chain[i+1]
                    child['joint_origin_translation'] = [v*new[i]/old[i] for v in child['joint_origin_translation']]
    # Axis, palm chassis and pads are applied by the phenotype compiler; they
    # are explicit metadata, never mistaken for bone radius or friction genes.
    return hand, dict(code=list(code), genes=dict(zip(NAMES, code)), baseline_sha256=digest(baseline),
                      palm_width_m=palm_width_m*(1+.1*code[0]),
                      thumb_axis_rotation_rad=math.radians(5)*code[6],
                      pad_width_scale=1+.1*code[7])


def initial_population(size=8, seed=20261024):
    if not 1 <= size <= 17:
        raise ValueError("Initial local neighborhood supports 1..17 candidates")
    neighbors = []
    for i in range(8):
        for level in (-1, 1):
            row = [0]*8; row[i] = level; neighbors.append(tuple(row))
    random.Random(seed).shuffle(neighbors)
    return [BASELINE] + neighbors[:size-1]
