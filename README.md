# Federated Learning using CIFAR-10
Implementing federated learning on IoT devices using the CIFAR-10 dataset

## Abstract
This is the final project for the Intelligent IoT Systems course taught by Professor Woonghee Lee at Hansung University. Federated learning is emerging as a crucial technology among the numerous users who own IoT devices. It not only ensures personal information protection and data privacy but also serves as a key method for developing advanced AI models based on diverse user data. In this project, the objective is to develop a system that establishes a federated learning environment between a central server and IoT devices, enabling continuous updates and improvements of AI models without transmitting the data stored on individual devices to the central server.

---

## Class Imbalance Problem


train_partition0 class counts: Counter({0: 2535, 1: 352, 2: 280, 3: 4445, 4: 303, 5: 4086, 6: 4474, 7: 1554, 8: 1125, 9: 5000})

train_partition1 class counts: Counter({0: 2465, 1: 4648, 2: 4720, 3: 555, 4: 4697, 5: 914, 6: 526, 7: 3446, 8: 3875, 9: 0})

test_dataset class counts: Counter({3: 1000, 8: 1000, 0: 1000, 6: 1000, 1: 1000, 9: 1000, 5: 1000, 7: 1000, 4: 1000, 2: 1000})

-> We can check Class Imbalance Problem. We tried to use focal-loss or weightening on small amount data. But, it wasn't work.


## My Notion

https://rustic-flavor-e48.notion.site/Federated-Learning-using-CIFAR-10-14fe2f945c638064b96cde9255a904d2?pvs=4

## Result of final project
Our result is team 1.

