# These units have roughly mean 0 and standard deviation 1
VERY_LOW = -3
LOW = -2
RATHER_LOW = -1.25
MEDIUM_LOW = -0.75
MEDIUM = 0
MEDIUM_HIGH = 0.75
RATHER_HIGH = 1.25
HIGH = 2
VERY_HIGH = 3

# see https://en.wikipedia.org/wiki/List_of_cloud_types

CLOUD_TYPES = {
    'Noctilucent clouds': {
        'Type 1': {
            'transparency': VERY_HIGH,
            'sharp edges': VERY_LOW,
            'elongation': MEDIUM_HIGH,
            'solidity': RATHER_LOW,
        },
        'Type 2': {
            'A': {
                'transparency': HIGH,
                'sharp edges': LOW,
                'elongation': HIGH,
                'solidity': MEDIUM_LOW,
                'rectangularity': RATHER_HIGH,
            },
            'B': {
                'transparency': HIGH,
                'sharp edges': MEDIUM,
                'elongation': HIGH,
                'solidity': MEDIUM_LOW,
                'rectangularity': RATHER_HIGH,
            }
        },
        'Type 3': {
            'A': {
                'transparency': HIGH,
                'sharp edges': MEDIUM,
                'elongation': MEDIUM,
                'rectangularity': HIGH,
                'solidity': HIGH,
            },
            'B': {
                'transparency': HIGH,
                'sharp edges': MEDIUM,
                'elongation': MEDIUM,
                'rectangularity': HIGH,
                'solidity': MEDIUM,
            },
        },
        'Type 4': {
            'transparency': LOW,
            'sharp edges': MEDIUM,
            'elongation': LOW,
            'roundness': VERY_HIGH,
            'solidity': HIGH,
        },
    },
    'Polar stratospheric clouds': {
        'Type 1': {
            'transparency': HIGH,
            'sharp edges': LOW,
            'mean r': HIGH,
            'std r': HIGH,
        },
        'Type 2': {
            'transparency': HIGH,
            'sharp edges': RATHER_LOW,
            'mean r': HIGH,
            'std r': HIGH,
        },
    },
    'Tropospheric clouds': {
        'High level': {
            'Genus cirrus': {
                'transparency': MEDIUM_HIGH,
                'sharp edges': RATHER_HIGH,
                'std': HIGH,
                'solidity': LOW,
            },
            'Genus cirrocumulus': {
                'transparency': MEDIUM_LOW,
                'sharp edges': RATHER_HIGH,
                'solidity': HIGH,
                'convexity': RATHER_HIGH,
                'roundness': HIGH,
            },
            'Genus cirrostratus': {
                'transparency': RATHER_LOW,
                'sharp edges': RATHER_LOW,
                'solidity': HIGH,
                'convexity': RATHER_HIGH,
                'roundness': HIGH,
            },
        },
        'Mid level': {
            'Genus altocumulus': {
                'Stratiformis': {
                    'transparency': RATHER_LOW,
                    'sharp edges': HIGH,
                },
                'Lenticularis': {
                    'transparency': LOW,
                    'sharp edges': HIGH,
                    'roundness': VERY_HIGH,
                },
                'Volutus': {
                    'transparency': RATHER_LOW,
                    'sharp edges': RATHER_HIGH,
                    'elongation': HIGH,
                },
                'Castellanus': {
                    'transparency': LOW,
                    'convexity': MEDIUM,
                    'std': MEDIUM_LOW,
                },
                'Floccus': {
                    'transparency': MEDIUM,
                    'std': HIGH,
                    'convexity': MEDIUM,
                    'solidity': MEDIUM_HIGH,
                },
            },
            'Genus altostratus': {
                'transparency': MEDIUM_HIGH,
                'solidity': HIGH,
            },
        },
        'Towering vertical': {
            'Genus cumulonimbus': {
                'transparency': VERY_LOW,
                'sharp edges': VERY_HIGH,
                'std': RATHER_LOW,
            },
            'Genus cumulus': {
                'transparency': VERY_LOW,
                'sharp edges': HIGH,
                'std': LOW,
            },
        },
        'moderate vertical': {
            'Genus nimbostratus': {
                'transparency': LOW,
                'sharp edges': MEDIUM,
            },
        },
        'Low level': {
            'Genus stratocumulus': {
                'transparency': RATHER_LOW,
                'sharp edges': MEDIUM,
                'std': RATHER_HIGH,
            },
            'Genus stratus': {
                'transparency': VERY_HIGH,
                'sharp edges': VERY_LOW,
                'convexity': HIGH,
            },
        }
    },
}
