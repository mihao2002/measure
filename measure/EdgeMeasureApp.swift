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
    
    // Visual elements for edge detection feedback
    private var edgeHighlightEntity: ModelEntity?
    private var measurementLineEntity: ModelEntity?
    private var centerIndicatorEntity: ModelEntity?

    required init(frame: CGRect) {
        super.init(frame: frame)
        self.setupFrameProcessing()
        self.setupVisualElements()
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
                
                let edgeFound = self.findEdgeNearCenter(observations: observations, center: center)
                if edgeFound {
                    DispatchQueue.main.async {
                        self.raycastLength(at: CGPoint(x: self.bounds.midX, y: self.bounds.midY))
                    }
                }
                
                // Update visual feedback
                self.updateVisualFeedback(edgeFound: edgeFound)
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
        
        // Update visual feedback with measurement points
        updateVisualFeedback(edgeFound: true, measurementPoints: (p1, p2))
    }

    // MARK: - Visual Feedback Methods
    private func setupVisualElements() {
        // Create center indicator (crosshair)
        let centerMaterial = SimpleMaterial(color: .red, isMetallic: false)
        let centerMesh = MeshResource.generateBox(size: 0.01)
        centerIndicatorEntity = ModelEntity(mesh: centerMesh, materials: [centerMaterial])
        
        // Create measurement line
        let lineMaterial = SimpleMaterial(color: .yellow, isMetallic: false)
        let lineMesh = MeshResource.generateBox(size: [0.1, 0.002, 0.002])
        measurementLineEntity = ModelEntity(mesh: lineMesh, materials: [lineMaterial])
        
        // Add to scene
        if let centerIndicator = centerIndicatorEntity {
            scene.addAnchor(AnchorEntity(world: [0, 0, -0.5]))
            scene.anchors.first?.addChild(centerIndicator)
        }
    }
    
    private func updateVisualFeedback(edgeFound: Bool, measurementPoints: (SIMD3<Float>, SIMD3<Float>)? = nil) {
        DispatchQueue.main.async {
            if edgeFound {
                // Show measurement line
                if let points = measurementPoints {
                    self.showMeasurementLine(from: points.0, to: points.1)
                }
                
                // Highlight center indicator
                self.centerIndicatorEntity?.model?.materials = [SimpleMaterial(color: .green, isMetallic: false)]
            } else {
                // Reset center indicator
                self.centerIndicatorEntity?.model?.materials = [SimpleMaterial(color: .red, isMetallic: false)]
                
                // Hide measurement line
                self.measurementLineEntity?.removeFromParent()
            }
        }
    }
    
    private func showMeasurementLine(from startPoint: SIMD3<Float>, to endPoint: SIMD3<Float>) {
        // Remove existing line
        measurementLineEntity?.removeFromParent()
        
        // Calculate line properties
        let direction = normalize(endPoint - startPoint)
        let distance = simd_distance(startPoint, endPoint)
        let midPoint = (startPoint + endPoint) / 2
        
        // Create line mesh
        let lineMaterial = SimpleMaterial(color: .yellow, isMetallic: false)
        let lineMesh = MeshResource.generateBox(size: [distance, 0.002, 0.002])
        let lineEntity = ModelEntity(mesh: lineMesh, materials: [lineMaterial])
        
        // Position and orient the line
        lineEntity.position = midPoint
        lineEntity.look(at: endPoint, from: midPoint, relativeTo: nil)
        
        // Add to scene
        if let anchor = scene.anchors.first {
            anchor.addChild(lineEntity)
            measurementLineEntity = lineEntity
        }
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


