from capsnet import *

img_size = 17
in_channel = 1024
batch_size = 3

model = CapsNet(image_size=img_size, in_channel=in_channel).cuda()

# print(model)
# summary(model, (in_channel, img_size, img_size))
input = torch.randn(batch_size, in_channel, img_size, img_size).cuda()
output = model(input)
print(f'input_size :{input.size()}')
print(f'output_size:{output.size()}')
print(output)

# 13x13x1024
# 26x26x512
# 52x52x256

