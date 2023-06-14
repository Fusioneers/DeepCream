"""This module contains the cloud types used by the classification"""

# These values are measured in standard deviation units and resemble the
# different intensities of the parameters a cloud can have.
VERY_LOW = -3
LOW = -2
RATHER_LOW = -1.25
MEDIUM_LOW = -0.75
MEDIUM = 0
MEDIUM_HIGH = 0.75
RATHER_HIGH = 1.25
HIGH = 2
VERY_HIGH = 3

# for a description of the different types see
# https://en.wikipedia.org/wiki/List_of_cloud_types

# This dictionary contains the cloud types. It has the following structure:
# CLOUD_TYPES:
#   group: (grouped by altitude)
#       type:
#           parameter: value
#
# The values in the types are in the class Classification against a specific
# cloud compared.
CLOUD_TYPES = {
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
            'sharp edges': HIGH,
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
            'transparency': RATHER_LOW,
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
}
