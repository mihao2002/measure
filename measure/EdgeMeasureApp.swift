// EdgeMeasureApp.swift
// MVP Prototype: Detect and measure an edge using ARKit + Vision + SwiftUI

import SwiftUI
import RealityKit
import ARKit
import Vision
import Combine

// MARK: - Shared ARView Singleton
class SharedARView: ARView {
    static let instance: SharedARView = {
        let arView = SharedARView(frame: .zero)
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        config.sceneReconstruction = .mesh
        arView.session.run(config)
        return arView
    }()

    private var framePublisher: AnyCancellable?
    private let visionQueue = DispatchQueue(label: "vision-edge-detection")
    var onEdgeLengthUpdate: ((Float) -> Void)?

    required init(frame: CGRect) {
        super.init(frame: frame)
        self.setupFrameProcessing()
    }

    required init?(coder: NSCoder) { fatalError() }

    private func setupFrameProcessing() {
        framePublisher = Timer.publish(every: 1.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in self?.processCurrentFrame() }
    }

    private func processCurrentFrame() {
        guard let frame = session.currentFrame else { return }

        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])

        let request = VNDetectContoursRequest()
        request.detectsDarkOnLight = true

        visionQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                try handler.perform([request])
                guard let observations = request.results?.first as? VNContoursObservation else { return }

                let center = CGPoint(x: 0.5, y: 0.5)
                
                if self.findEdgeNearCenter(observations: observations, center: center) {
                    DispatchQueue.main.async {
                        self.raycastLength(at: CGPoint(x: self.bounds.midX, y: self.bounds.midY))
                    }
                }
            } catch {
                print("Vision error: \(error)")
            }
        }

    }

    private func findEdgeNearCenter(observations: VNContoursObservation, center: CGPoint) -> Bool {
        // Check if any contour has points near the center
        for contourIndex in 0..<observations.topLevelContours.count {
            let contour = observations.topLevelContours[contourIndex]
            if hasPointNearCenter(contour: contour, center: center) {
                return true
            }
        }
        return false
    }
    
    private func hasPointNearCenter(contour: VNContour, center: CGPoint) -> Bool {
        // Get the normalized points for this contour
        let points = contour.normalizedPoints
        return points.contains(where: { point in
            return self.isPointNearCenter(point: point, center: center)
        })
    }
    
    private func isPointNearCenter(point: simd_float2, center: CGPoint) -> Bool {
        return abs(point.x - Float(center.x)) < 0.05 && abs(point.y - Float(center.y)) < 0.05
    }

    private func raycastLength(at screenPoint: CGPoint) {
        let results = raycast(from: screenPoint, allowing: .estimatedPlane, alignment: .any)
        guard let result1 = results.first else { return }

        let offset = CGPoint(x: screenPoint.x + 10, y: screenPoint.y)
        let results2 = raycast(from: offset, allowing: .estimatedPlane, alignment: .any)
        guard let result2 = results2.first else { return }

        let p1 = result1.worldTransform.translation
        let p2 = result2.worldTransform.translation
        let distance = simd_distance(p1, p2)

        onEdgeLengthUpdate?(distance)
    }
}

extension simd_float4x4 {
    var translation: SIMD3<Float> {
        return [columns.3.x, columns.3.y, columns.3.z]
    }
}

struct ARViewContainer: UIViewRepresentable {
    @Binding var edgeLength: Float

    func makeUIView(context: Context) -> SharedARView {
        let arView = SharedARView.instance
        arView.onEdgeLengthUpdate = { length in
            self.edgeLength = length
        }
        return arView
    }

    func updateUIView(_ uiView: SharedARView, context: Context) {}
}


