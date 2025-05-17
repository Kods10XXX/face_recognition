import asyncio
import websockets
import cv2
import numpy as np

async def receive_video_feed():
    uri = "ws://localhost:8000/video_feed"
    async with websockets.connect(uri) as websocket:
        while True:
            
            # Receive JSON data first
            result = await websocket.recv()
            print("Recognition result:", result)
            
            # Then receive binary image data
            image_data = await websocket.recv()
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow('Video Feed', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

asyncio.get_event_loop().run_until_complete(receive_video_feed())
cv2.destroyAllWindows()