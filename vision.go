package main

import (
	"context"
	"fmt"
	"strings"

	vision "cloud.google.com/go/vision/apiv1"
)

// Annotate an image file with face based on Cloud Vision API, return score and error if exists.
func annotateFace(uri string) (float32, error) {
	// Creates a client
	ctx := context.Background()
	client, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		return 0.0, err
	}
	defer client.Close()

	image := vision.NewImageFromURI(uri)
	annotations, err := client.DetectFaces(ctx, image, nil, 1)
	if err != nil {
		return 0.0, err
	}
	if len(annotations) == 0 {
		fmt.Println("No faces found.")
		return 0.0, nil
	}
	return annotations[0].DetectionConfidence, nil
}

// Annotate an image file with food / exervise based on Cloud Vision API, return score (food), error, score(exercise), error
func annotateFoodExcercise(uri string) (float32, float32, error) {
	ctx := context.Background()
	client, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		return 0.0, 0.0, err
	}
	defer client.Close()

	image := vision.NewImageFromURI(uri)
	annotations, err := client.DetectLabels(ctx, image, nil, 5)
	if err != nil || (len(annotations) == 0) {
		fmt.Println("No labels found or error!.")
		return 0.0, 0.0, err
	}

	var food, exercise = 0.0, 0.0

	for _, annotation := range annotations {
		if annotation.Description == "Food" || annotation.Description == "food" {
			food = 1.0
		}
		if strings.Contains(annotation.Description, "Exercise") ||
			strings.Contains(annotation.Description, "Fitness") ||
			strings.Contains(annotation.Description, "exercise") ||
			strings.Contains(annotation.Description, "fitness") {
			exercise = 1.0
		}
	}
	return float32(food), float32(exercise), nil
}
