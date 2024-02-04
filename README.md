# A Study on Transfer Learning-Based Investigation of Backdoor Attack Vulnerabilities in 3D Point Cloud Classifiers
PointNet and PointNet++ codes were downloaded from the link below.  
https://github.com/yanx27/Pointnet_Pointnet2_pytorch

## Abstract (Korean)
딥러닝 기술의 발전과 함께 전이학습 기반의 포인트 클라우드 분류 모델들이 등장하고 있다. 하지만, 전이학습이 백도어 공격에 취약함이 밝혀짐에 따라 전이학습 기반의 포인트 클라우드 분류 모델에 대한 백도어 공격 위험성이 대두되고 있다. 본 논문에서는 전이학습 기반의 포인트 클라우드 분류 모델이 백도어 공격에 얼마나 취약한지를 실험적으로 분석한다. 구체적으로, 포인트 클라우드 분류기에 백도어를 삽입하는 사전학습과 깨끗한 데이터셋으로 미세조정하는 단계를 통해 실험을 수행한다. 미세조정 단계에서 오염되지 않은 깨끗한 데이터로 학습을 수행할 경우, 백도어 활성화를 방지할 수 있음을 확인한다. 이를 통해, 깨끗한 데이터셋을 통한 미세조정이 백도어 공격 방어에 효과적임을 시사한다. 
키워드 : 전이학습, 포인트 클라우드 분류, 백도어 공격

## Abstract (English)
With the advancement of deep learning technology, transfer learning-based point cloud classification models have emerged. However, the vulnerability of transfer learning to backdoor attacks has been highlighted, raising concerns about the security risks for point cloud classification models based on transfer learning. In this paper, we experimentally analyze how vulnerable transfer learning-based point cloud classification models are to backdoor attacks. Specifically, the analysis methodology involves pre-training the point cloud classifiers with backdoor insertion, followed by fine-tuning them with a clean dataset. From the experimental results, it is observed that fine-tuning with a clean dataset during the fine-tuning phase can prevent the activation of the backdoor. This suggests that fine-tuning with a clean dataset can be an effective defense mechanism against backdoor attacks.
Key Words : Transfer Learning, Point Cloud Classification, Backdoor Attack

## Experiment Results
![image](https://github.com/parkie0517/Transfer-Learning-Based-Backdoor-Attack-Vulnerabilities-in-3D-Point-Cloud-Classifiers/assets/80407632/d0a94b6b-1077-405e-a53f-cab27448a8bb)  

![image](https://github.com/parkie0517/Transfer-Learning-Based-Backdoor-Attack-Vulnerabilities-in-3D-Point-Cloud-Classifiers/assets/80407632/f7621355-8dc2-4dfe-8135-da97d4e74700)  

![image](https://github.com/parkie0517/Transfer-Learning-Based-Backdoor-Attack-Vulnerabilities-in-3D-Point-Cloud-Classifiers/assets/80407632/7a190ea0-e70c-4e3b-bb33-a1d5d55e18ea)  
