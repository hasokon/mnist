package mnist

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

func ReadLabel(reader *bufio.Reader) (int, error) {
	b, err := reader.ReadByte()
	if err != nil {
		return 0, err
	}

	return int(b), nil
}

type MNIST struct {
	TrainImages      [][]uint8
	TrainLabels      []int
	TestImages       [][]uint8
	TestLabels       []int
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
	mnist.TrainLabels = make([]int, mnist.TrainDataSize)
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
	mnist.TestLabels = make([]int, mnist.TestDataSize)
	for i := 0; i < mnist.TestDataSize; i++ {
		labeldata, err := ReadLabel(tlreader)
		if err != nil {
			fmt.Println(err.Error())
			return nil, err
		}
		mnist.TestLabels[i] = labeldata
	}

	// fmt.Println(mnist.TrainLabels)
	return mnist, nil
}

func (m *MNIST) GetTrainImagesFloat64() [][]float64 {
	trainimagesFloat64 := make([][]float64, m.TrainDataSize)
	imageSize := m.TrainImageHeight * m.TrainImageWidth
	for i := 0; i < m.TrainDataSize; i++ {
		trainimagesFloat64[i] = make([]float64, imageSize)
		for j := 0; j < imageSize; j++ {
			trainimagesFloat64[i][j] = float64(m.TrainImages[i][j])
		}
	}

	return trainimagesFloat64
}

func (m *MNIST) GetTestImagesFloat64() [][]float64 {
	testimagesFloat64 := make([][]float64, m.TestDataSize)
	imageSize := m.TestImageHeight * m.TestImageWidth
	for i := 0; i < m.TestDataSize; i++ {
		testimagesFloat64[i] = make([]float64, imageSize)
		for j := 0; j < imageSize; j++ {
			testimagesFloat64[i][j] = float64(m.TestImages[i][j])
		}
	}

	return testimagesFloat64
}

func (m *MNIST) GetTrainLabelsOneHot() [][]float64 {
	trainlabels := make([][]float64, m.TrainDataSize)
	for i := 0; i < m.TrainDataSize; i++ {
		trainlabels[i] = make([]float64, labelSize)
		trainlabels[i][m.TrainLabels[i]] = 1.0
	}

	return trainlabels
}

func (m *MNIST) GetTestLabelsOneHot() [][]float64 {
	testlabels := make([][]float64, m.TestDataSize)
	for i := 0; i < m.TestDataSize; i++ {
		testlabels[i] = make([]float64, labelSize)
		testlabels[i][m.TestLabels[i]] = 1.0
	}

	return testlabels
}

func flat(t [][]float64) []float64 {
	l0 := len(t)
	l1 := len(t[0])
	ans := make([]float64, 0, l0*l1)

	for i := 0; i < l0; i++ {
		ans = append(ans, t[i]...)
	}

	return ans
}

func (m *MNIST) GetDataForNN() (trainimages, trainlabels, testimages, testlabels []float64) {
	trainimages = flat(m.GetTrainImagesFloat64())
	trainlabels = flat(m.GetTrainLabelsOneHot())
	testimages = flat(m.GetTestImagesFloat64())
	testlabels = flat(m.GetTestLabelsOneHot())

	return
}
