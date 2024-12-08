#Data load
!pip install gdown==4.6.0
!gdown https://drive.google.com/uc?id=1CPX6pilBUv6XbCootSvnJ1ep-3hHMJwr

!unzip /content/Animals90.zip
transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor()])
transform_val = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor()])
transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor()])
train_DS = torchvision.datasets.ImageFolder(root = "/content/Animals90/train_DS", transform = transform_train)
val_DS = torchvision.datasets.ImageFolder(root = "/content/Animals90/val_DS", transform = transform_val)
test_DS = torchvision.datasets.ImageFolder(root = "/content/Animals90/test_DS", transform = transform_test)
train_DL = DataLoader(train_DS, batch_size = BATCH_SIZE, shuffle = True)
val_DL = DataLoader(val_DS, batch_size = BATCH_SIZE, shuffle = True)
test_DL = DataLoader(test_DS, batch_size = BATCH_SIZE, shuffle = True)
