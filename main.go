package main

import (
	"image"
	"image/color"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten"
)

const (
	width     = 640
	height    = 480
	intervalH = 3
	intervalW = 2
)

func GetUpdate() func(*ebiten.Image) error {
	mnist, _ := InitMNIST()

	// imgwidth := mnist.TrainImageWidth
	// imgheight := mnist.TrainImageHeight
	// numOfImages := mnist.TrainDataSize
	// images := mnist.TrainImages

	imgwidth := mnist.TestImageWidth
	imgheight := mnist.TestImageHeight
	numOfImages := mnist.TestDataSize
	images := mnist.TestImages

	widthNum := width / (imgwidth + intervalW*2)
	heightNum := height / (imgheight + intervalH*2)

	displayImages := make([][]*ebiten.Image, heightNum)

	rand.Seed(time.Now().UnixNano())
	index := rand.Perm(numOfImages)

	for i := 0; i < heightNum; i++ {
		displayImages[i] = make([]*ebiten.Image, widthNum)
		for j := 0; j < widthNum; j++ {
			displayImages[i][j], _ = ebiten.NewImageFromImage(
				&image.Gray{
					Pix:    images[index[i*widthNum+j]],
					Stride: imgwidth,
					Rect:   image.Rect(0, 0, imgwidth, imgheight),
				}, ebiten.FilterLinear)
		}
	}

	return func(screen *ebiten.Image) error {

		screen.Clear()
		screen.Fill(color.White)

		op := &ebiten.DrawImageOptions{}
		op.GeoM.Translate(float64(intervalW), float64(intervalH))

		for i := 0; i < heightNum; i++ {
			for j := 0; j < widthNum; j++ {
				screen.DrawImage(displayImages[i][j], op)
				op.GeoM.Translate(float64(intervalW*2+imgwidth), 0)
			}
			op.GeoM.Translate(float64((intervalW*2+imgwidth)*widthNum*-1), float64(intervalH*2+imgheight))
		}
		return nil
	}
}

func main() {
	ebiten.Run(GetUpdate(), width, height, 2.0, "MNIST Images")
}
