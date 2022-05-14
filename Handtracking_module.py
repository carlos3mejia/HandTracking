import cv2
import mediapipe as mp
import time

class Hands_Detector():

    def __init__(self, mode=False, max_hands=2, complexity=1, detection_confidence=0.5, track_confidence=0.5,
                 lm_change_color=False):
        '''
        Description:
        ------------
        Initialize parameters of Hand detector

        Parameters
        ----------
        mode(boolean) :
            - False hand detection runs in the first input image
            - True hand detection runs for every input image
        max_hands(int): Maximum number of hands to detect
        complexity(complexity)(int): Complexity of the hand landmark model: 0 or 1. Landmark accuracy as well as
                                     inference latency generally go up with the model complexity.
        detection_confidence(float): Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection
                             to be considered successful.
        trackCon(float): - Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks
                           to be considered tracked successfully, or otherwise hand detection will be invoked
                           automatically on the next input image.
                         - Higher value can increase robustness of the solution, at the expense of a higher latency.
                         - Ignored if static_image_mode is true, where hand detection simply runs on every image.
        self.change_color (boolean):
            - False No color change in the hand landmarks
            - True Color change in the hand landmarks (Default "yellow")

        Returns:
        --------

        self.mpHands : load "solutions.hands" methods from mediapipe
        self.hands : load method to detect hands using the input parameters
        self.mpDraw : load method for drawing landmarks from mediapipe

        '''
        # Default size and color of the hand landmarks if lm_change_color is True
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        # Default size and color variable for the hand landmarks if lm_change_color is True
        self.change_color = lm_change_color
        self.lm_size = 15
        self.lm_color = (0, 255, 255)
        self.lm_pixel = []


        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.detection_confidence, self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def draw_landmarks(self, imagen, draw=True):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(imagen, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if self.change_color & (len(self.lm_pixel) != 0):
                    for coordinates in self.lm_pixel:
                        print(coordinates)
                        cv2.circle(imagen, coordinates[1:3], self.lm_size, self.lm_color, cv2.FILLED)

    def find_hands(self, imagen, draw=True):
        '''
        Description:
        ------------
        Find the hand in the image

        Parameters
        ----------
        img (numpy.ndarray):
        Captured image from OpenCV

        draw (boolean):
        - True: Draw hand landmarks.
        - False: Don't draw hand landmarks.

        Returns
        -------
        img (numpy.ndarray):
        Returns the same image with or without hand landmarks if draw is equal to True or False, respectively
        '''
        # Transform image from BGR to RGB to detect hands and identify coordinates of 21 landmark of each hand shown
        # in the picture
        imagen_RGB = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imagen_RGB)
        # Draw the hand landmarks
        self.draw_landmarks(imagen, draw)
        return imagen

    def find_hand_landmarks(self, imagen, hand_id=0):
        '''
        Description
        -----------
        Transform landmarks of a specific hand

        Parameters
        ----------
        img (numpy.ndarray):
        Captured image from OpenCV

        handNum (int):
        Hand ID that you want to detect

        Returns:
        -------
        self.lm_pixel (list):
        List groups where each element contains information of the landmark identification (id) and position in pixels (px,py) [id, px, py]
        '''
        pixel = []
        if self.results.multi_hand_landmarks:
            hand_landmark = self.results.multi_hand_landmarks[hand_id]
            for lm_id, landmark in enumerate(hand_landmark.landmark):
                    height, width, color = imagen.shape
                    pixel_x, pixel_y = int(landmark.x*width), int(landmark.y*height)
                    pixel.append([lm_id, pixel_x, pixel_y])
        self.lm_pixel = pixel
        return self.lm_pixel


def main():
    past_time = 0

    # Capture image with the camera
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("Test_image", cv2.WINDOW_KEEPRATIO)
    # Load object for detecting the hands
    detector = Hands_Detector()

    while cv2.getWindowProperty("Test_image", cv2.WND_PROP_VISIBLE) >= 1:
        #read image
        success, imagen = capture.read()
        imagen_hands = detector.find_hands(imagen)
        lm_pixels = detector.find_hand_landmarks(imagen_hands)


        #Print list that contain index, pixel_x, and pixel_y of the eight landmark
        if len(lm_pixels) != 0:
            print(lm_pixels [8])


        # Calculate Number Frames per Second
        current_time = time.time()
        frames_per_second = 1 / (current_time - past_time)
        past_time = current_time

        # Display image
        cv2.putText(imagen, str(int(frames_per_second)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Test_image", imagen)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()