package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
)

const (
	labelSize = 10
)

func ReadBytes(len int, reader *bufio.Reader) ([]uint8, int, error) {
	readdata := make([]uint8, 0, len)
	count := 0
	for i := 0; i < len; i++ {
		b, err := reader.ReadByte()
		if err != nil {
			return readdata, count, err
		}
		readdata = append(readdata, uint8(b))
		count++
	}
	return readdata, count, nil
}

func ReadLabel(reader *bufio.Reader) ([]bool, error) {
	labeldata := make([]bool, labelSize)

	b, err := reader.ReadByte()
	if err != nil {
		return labeldata, err
	}

	labeldata[int(b)-1] = true

	return labeldata, nil
}

type MNIST struct {
	TrainImages      [][]uint8
	TrainLabels      [][]bool
	TestImages       [][]uint8
	TestLabels       [][]bool
	TrainDataSize    int
	TrainImageWidth  int
	TrainImageHeight int
	TestDataSize     int
	TestImageWidth   int
	TestImageHeight  int
}

func InitMNIST() (*MNIST, error) {
	mnist := &MNIST{}

	trainimages, err := os.Open("train-images-idx3-ubyte")
	if err != nil {
		return nil, err
	}
	defer trainimages.Close()

	trainlabel, err := os.Open("train-labels-idx1-ubyte")
	if err != nil {
		return nil, err
	}
	defer trainlabel.Close()

	testimages, err := os.Open("t10k-images-idx3-ubyte")
	if err != nil {
		return nil, err
	}
	defer testimages.Close()

	testlabel, err := os.Open("t10k-labels-idx1-ubyte")
	if err != nil {
		return nil, err
	}
	defer testlabel.Close()

	// Train Images
	tireader := bufio.NewReader(trainimages)
	ReadBytes(4, tireader) //Magick Number
	numOfImages, _, _ := ReadBytes(4, tireader)
	numOfRows, _, _ := ReadBytes(4, tireader)
	numOfCols, _, _ := ReadBytes(4, tireader)

	rows := binary.BigEndian.Uint32(numOfRows)
	cols := binary.BigEndian.Uint32(numOfCols)

	mnist.TrainDataSize = int(binary.BigEndian.Uint32(numOfImages))
	mnist.TrainImageWidth = int(cols)
	mnist.TrainImageHeight = int(rows)
	imagesize := int(rows * cols)

	mnist.TrainImages = make([][]uint8, mnist.TrainDataSize)
	for i := 0; i < mnist.TrainDataSize; i++ {
		img, size, err := ReadBytes(imagesize, tireader)
		if err != nil {
			fmt.Printf("Read Size:%d, Err:%s\n", size, err.Error())
			return nil, err
		}
		mnist.TrainImages[i] = img
	}

	// Train Label
	tilreader := bufio.NewReader(trainlabel)
	ReadBytes(8, tilreader)
	mnist.TrainLabels = make([][]bool, mnist.TrainDataSize)
	for i := 0; i < mnist.TrainDataSize; i++ {
		labeldata, err := ReadLabel(tilreader)
		if err != nil {
			fmt.Println(err.Error())
			return nil, err
		}
		mnist.TrainLabels[i] = labeldata
	}

	// Test Images
	treader := bufio.NewReader(testimages)
	ReadBytes(4, treader) //Magick Number
	numOfImages, _, _ = ReadBytes(4, treader)
	numOfRows, _, _ = ReadBytes(4, treader)
	numOfCols, _, _ = ReadBytes(4, treader)

	rows = binary.BigEndian.Uint32(numOfRows)
	cols = binary.BigEndian.Uint32(numOfCols)

	mnist.TestDataSize = int(binary.BigEndian.Uint32(numOfImages))
	mnist.TestImageWidth = int(cols)
	mnist.TestImageHeight = int(rows)
	imagesize = int(rows * cols)

	mnist.TestImages = make([][]uint8, mnist.TestDataSize)
	for i := 0; i < mnist.TestDataSize; i++ {
		img, size, err := ReadBytes(imagesize, treader)
		if err != nil {
			fmt.Printf("Read Size:%d, Err:%s\n", size, err.Error())
			return nil, err
		}
		mnist.TestImages[i] = img
	}

	// Test Label
	tlreader := bufio.NewReader(testlabel)
	ReadBytes(8, tlreader)
	mnist.TestLabels = make([][]bool, mnist.TestDataSize)
	for i := 0; i < mnist.TestDataSize; i++ {
		labeldata, err := ReadLabel(tlreader)
		if err != nil {
			fmt.Println(err.Error())
			return nil, err
		}
		mnist.TestLabels[i] = labeldata
	}

	return mnist, nil
}
