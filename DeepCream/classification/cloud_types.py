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


# TODO document
CLOUD_TYPES = {
    # 'Noctilucent clouds': {
    #     'Type 1': {
    #         'A': {
    #             'transparency': VERY_HIGH,
    #             'sharp edges': VERY_LOW,
    #             'elongation': MEDIUM_HIGH,
    #             'solidity': RATHER_LOW,
    #         },
    #     },
    #     'Type 2': {
    #         'A': {
    #             'transparency': HIGH,
    #             'sharp edges': LOW,
    #             'elongation': HIGH,
    #             'solidity': MEDIUM_LOW,
    #             'rectangularity': RATHER_HIGH,
    #         },
    #         'B': {
    #             'transparency': HIGH,
    #             'sharp edges': MEDIUM,
    #             'elongation': HIGH,
    #             'solidity': MEDIUM_LOW,
    #             'rectangularity': RATHER_HIGH,
    #         },
    #     },
    #     'Type 3': {
    #         'A': {
    #             'transparency': HIGH,
    #             'sharp edges': MEDIUM,
    #             'elongation': MEDIUM,
    #             'rectangularity': HIGH,
    #             'solidity': HIGH,
    #         },
    #         'B': {
    #             'transparency': HIGH,
    #             'sharp edges': MEDIUM,
    #             'elongation': MEDIUM,
    #             'rectangularity': HIGH,
    #             'solidity': MEDIUM,
    #         },
    #     },
    #     'Type 4': {
    #         'A': {
    #             'transparency': LOW,
    #             'sharp edges': MEDIUM,
    #             'elongation': LOW,
    #             'roundness': VERY_HIGH,
    #             'solidity': HIGH,
    #         },
    #     },
    # },
    # 'Polar stratospheric clouds': {
    #     'Type 1': {
    #         'A': {
    #             'transparency': HIGH,
    #             'sharp edges': LOW,
    #             'mean r': HIGH,
    #             'std r': HIGH,
    #         },
    #     },
    #     'Type 2': {
    #         'A': {
    #             'transparency': HIGH,
    #             'sharp edges': RATHER_LOW,
    #             'mean r': HIGH,
    #             'std r': HIGH,
    #         },
    #     },
    # },
    'Tropospheric clouds': {
        'High level': {
            'Genus cirrus': {
                'transparency': MEDIUM_HIGH,
                'sharp edges': HIGH,
                'std': HIGH,
                'solidity': LOW,
                'convexity': VERY_LOW,
            },
            'Genus cirrocumulus': {
                'transparency': MEDIUM_HIGH,
                'sharp edges': RATHER_HIGH,
                'solidity': HIGH,
                'convexity': RATHER_HIGH,
                'roundness': HIGH,
                'std': HIGH,
            },
            'Genus cirrostratus': {
                'transparency': HIGH,
                'sharp edges': LOW,
                'solidity': VERY_HIGH,
                'convexity': RATHER_HIGH,
            },
        },
        'Mid level': {
            'Genus altocumulus': {
                'transparency': MEDIUM,
                'std': VERY_HIGH,
                'convexity': LOW,
                'solidity': VERY_LOW,
            },
        },
        'Towering vertical': {
            'Genus cumulonimbus': {
                'transparency': VERY_LOW,
                'sharp edges': VERY_HIGH,
                'std': RATHER_LOW,
                'solidity': HIGH,
            },
            'Genus cumulus': {
                'transparency': LOW,
                'sharp edges': HIGH,
                'std': LOW,
            },
        },
        'moderate vertical': {
            'Genus nimbostratus': {
                'transparency': LOW,
                'sharp edges': LOW,
                'convexity': LOW,
                'std': VERY_LOW,
            },
        },
        'Low level': {
            'Genus stratocumulus': {
                'transparency': RATHER_HIGH,
                'sharp edges': MEDIUM_HIGH,
                'std': HIGH,
            },
            'Genus stratus': {
                'transparency': VERY_HIGH,
                'sharp edges': VERY_LOW,
                'convexity': HIGH,
            },
        },
    },
}
