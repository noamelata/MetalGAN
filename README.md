# MetalGAN
Generative adverserial neural network for metal band logo generation

## Overview
This project is a personal project to test my capabilities for implementing a GAN. 
I chose to try and generate metal band logos for several reasons.
1. I personally favour metal music.
2. I have abundunt data (~8k images). More would be better but I have found examples of impressive GANs trained on less than 1k images.
3. The logo images represent a good distibution for a GAN: 
  - they are mostly grey-scale (or monochrome and thus easily converted to grey-scale)
  - they retain meaning when resized
  - the can be quite hard to read, which makes the generator's job of creating believable images much easier

I am running most of the training on my personal computer, thus I have to keep to as little computing power as possible.
The implementation is mostly based on stackGAN.
currently this is a work in progress

## Results
first results with 64x64 were not to impressive, but proved that I am on the right track.

For the second version I added a higher definition block, and added progressive training of different layers.

## What to expect
I am currently tring to add a character based encoder, so that the GAN will be able to create a logo for a given name.
