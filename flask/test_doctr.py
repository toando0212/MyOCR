# List of known DocTR architectures
architectures = [
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "db_mobilenet_v3_small",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50"
]

print("Known DocTR architectures:")
for arch in architectures:
    print(f"- {arch}") 