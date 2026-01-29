// Frame.cpp

#include "Frame.hpp"

extern "C"
{
#include <libavutil/hwcontext.h>
}

using namespace nelux::error;

namespace nelux
{
/**
 * @brief Default constructor that allocates a new AVFrame.
 *
 * @throws FFException if AVFrame allocation fails.
 */
Frame::Frame() : frame(av_frame_alloc())
{
    NELUX_DEBUG("Frame constructor called: Allocating new AVFrame");
    if (!frame)
    {
        NELUX_ERROR("Failed to allocate AVFrame in default constructor");
        throw CxException("Failed to allocate AVFrame");
    }
    NELUX_INFO("Frame Class: CPU AVFrame allocated successfully in default constructor");
}

/**
 * @brief Constructor that takes an existing AVFrame pointer.
 *
 * @param frame Pointer to an existing AVFrame.
 * @throws FFException if the provided AVFrame pointer is null.
 */
Frame::Frame(AVFrame* frame) : frame(frame)
{
    NELUX_DEBUG("Frame constructor called with existing AVFrame pointer");
    if (!frame)
    {
        NELUX_ERROR("Null AVFrame provided to constructor");
        throw CxException("Null AVFrame provided");
    }
    NELUX_INFO("Frame Class: Frame initialized with existing AVFrame pointer");
}

Frame::Frame(AVBufferRef* hw_frames_ref) : frame(av_frame_alloc())
{
    NELUX_DEBUG("Hardware Frame constructor called: Allocating new AVFrame with "
                "hardware frames context");

    if (!frame)
    {
        NELUX_ERROR("Failed to allocate AVFrame in hardware constructor");
        throw CxException("Failed to allocate AVFrame");
    }

    // Assign the hardware frames context reference to the frame
    frame->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
    if (!frame->hw_frames_ctx)
    {
        NELUX_ERROR("Failed to set hardware frames context in hardware constructor");
        av_frame_free(&frame);
        throw CxException("Failed to set hardware frames context");
    }

    // Initialize format and dimensions from hw frames context
    AVHWFramesContext* frames_ctx =
        reinterpret_cast<AVHWFramesContext*>(frame->hw_frames_ctx->data);
    if (frames_ctx)
    {
        frame->format = frames_ctx->format;
        frame->width = frames_ctx->width;
        frame->height = frames_ctx->height;
    }

    // Allocate the frame buffers in hardware memory
    if (av_hwframe_get_buffer(frame->hw_frames_ctx, frame, 0) < 0)
    {
        NELUX_ERROR("Failed to allocate hardware frame buffer");
        av_frame_free(&frame);
        throw CxException("Failed to allocate hardware frame buffer");
    }

    NELUX_INFO("Frame Class: Hardware AVFrame allocated and initialized successfully");
}


/**
 * @brief Destructor that frees the AVFrame.
 */
Frame::~Frame()
{
    NELUX_DEBUG("Frame destructor called: Freeing AVFrame");
    av_frame_free(&frame);
    NELUX_DEBUG("Frame Class: AVFrame freed successfully in destructor");
}

/**
 * @brief Copy constructor that clones the AVFrame from another Frame.
 *
 * @param other The Frame object to copy from.
 * @throws FFException if AVFrame cloning fails.
 */
Frame::Frame(const Frame& other) : frame(av_frame_clone(other.frame))
{
    NELUX_DEBUG("Frame copy constructor called");
    if (!frame)
    {
        NELUX_ERROR("Failed to clone AVFrame in copy constructor");
        throw CxException("Failed to clone AVFrame");
    }
    NELUX_DEBUG("Frame Class: AVFrame cloned successfully in copy constructor");
}

/**
 * @brief Copy assignment operator that clones the AVFrame from another Frame.
 *
 * @param other The Frame object to copy from.
 * @return Reference to the assigned Frame object.
 * @throws FFException if AVFrame cloning fails.
 */
Frame& Frame::operator=(const Frame& other)
{
    NELUX_DEBUG("Frame copy assignment operator called");
    if (this != &other)
    {
        // Unreference and free the existing frame
        NELUX_DEBUG("Unreferencing and freeing existing AVFrame in copy assignment");
        av_frame_unref(frame);
        av_frame_free(&frame);

        // Clone the AVFrame from the other Frame
        NELUX_DEBUG("Cloning AVFrame from source Frame");
        frame = av_frame_clone(other.frame);
        if (!frame)
        {
            NELUX_ERROR("Failed to clone AVFrame during copy assignment");
            throw CxException("Failed to clone AVFrame during copy assignment");
        }
        NELUX_DEBUG("Frame Class: AVFrame cloned successfully in copy assignment");
    }
    else
    {
        NELUX_WARN("Self-assignment detected in copy assignment operator");
    }
    return *this;
}

/**
 * @brief Move constructor that transfers ownership of the AVFrame.
 *
 * @param other The Frame object to move from.
 */
Frame::Frame(Frame&& other) noexcept : frame(other.frame)
{
    NELUX_DEBUG("Frame move constructor called: Transferring AVFrame ownership");
    other.frame = nullptr;
    NELUX_DEBUG(
        "Frame Class: AVFrame ownership transferred successfully in move constructor");
}

/**
 * @brief Move assignment operator that transfers ownership of the AVFrame.
 *
 * @param other The Frame object to move from.
 * @return Reference to the assigned Frame object.
 */
Frame& Frame::operator=(Frame&& other) noexcept
{
    NELUX_DEBUG("Frame move assignment operator called");
    if (this != &other)
    {
        // Unreference and free the existing frame
        NELUX_DEBUG("Unreferencing and freeing existing AVFrame in move assignment");
        av_frame_unref(frame);
        av_frame_free(&frame);

        // Transfer ownership of the frame pointer
        frame = other.frame;
        other.frame = nullptr;
        NELUX_DEBUG("Frame Class: AVFrame ownership transferred successfully in move "
                    "assignment");
    }
    else
    {
        NELUX_WARN("Self-assignment detected in move assignment operator");
    }
    return *this;
}

/**
 * @brief Access the underlying AVFrame pointer.
 *
 * @return Pointer to the AVFrame.
 */
AVFrame* Frame::get() const
{
    NELUX_TRACE("Frame::get() called: Returning AVFrame pointer");
    return frame;
}

/**
 * @brief Get the width of the frame.
 *
 * @return Width in pixels.
 */
int Frame::getWidth() const
{
    NELUX_TRACE("Frame::getWidth() called: width = {}", frame->width);
    return frame->width;
}

/**
 * @brief Get the height of the frame.
 *
 * @return Height in pixels.
 */
int Frame::getHeight() const
{
    NELUX_TRACE("Frame::getHeight() called: height = {}", frame->height);
    return frame->height;
}

/**
 * @brief Get the pixel format of the frame.
 *
 * @return AVPixelFormat enumeration value.
 */
AVPixelFormat Frame::getPixelFormat() const
{
    NELUX_TRACE("Frame::getPixelFormat() called: format = {}",
                av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)));
    return static_cast<AVPixelFormat>(frame->format);
}

std::string Frame::getPixelFormatString() const
{
    std::string pix_fmt_str = av_get_pix_fmt_name(getPixelFormat())
                                  ? av_get_pix_fmt_name(getPixelFormat())
                                  : "Unknown";
    NELUX_TRACE("Frame::getPixelFormatString() called: format = {}", pix_fmt_str);
    return pix_fmt_str;
}

/**
 * @brief Get the data pointer for a specific plane.
 *
 * @param plane Plane index (0 for Y, 1 for U, 2 for V in YUV formats).
 * @return Pointer to the data of the specified plane.
 * @throws FFException if the plane index is out of range.
 */
uint8_t* Frame::getData(int plane) const
{
    NELUX_TRACE("Frame::getData() called with plane = {}", plane);
    if (plane < 0 || plane >= AV_NUM_DATA_POINTERS)
    {
        NELUX_ERROR("Invalid plane index in getData(): {}", plane);
        throw CxException("Invalid plane index: " + std::to_string(plane));
    }
    NELUX_DEBUG("Returning data pointer for plane {}: {}", plane,
                static_cast<void*>(frame->data[plane]));
    return frame->data[plane];
}

/**
 * @brief Get the line size for a specific plane.
 *
 * @param plane Plane index.
 * @return Line size in bytes.
 * @throws FFException if the plane index is out of range.
 */
int Frame::getLineSize(int plane) const
{
    NELUX_TRACE("Frame::getLineSize() called with plane = {}", plane);
    if (plane < 0 || plane >= AV_NUM_DATA_POINTERS)
    {
        NELUX_ERROR("Invalid plane index in getLineSize(): {}", plane);
        throw CxException("Invalid plane index: " + std::to_string(plane));
    }
    NELUX_DEBUG("Returning line size for plane {}: {}", plane, frame->linesize[plane]);
    return frame->linesize[plane];
}

/**
 * @brief Get the presentation timestamp of the frame.
 *
 * @return Presentation timestamp (PTS).
 */
int64_t Frame::getPts() const
{
    NELUX_TRACE("Frame::getPts() called: pts = {}", frame->pts);
    return frame->pts;
}

/**
 * @brief Set the presentation timestamp of the frame.
 *
 * @param pts New presentation timestamp.
 */
void Frame::setPts(int64_t pts)
{
    NELUX_TRACE("Frame::setPts() called: setting pts from {} to {}", frame->pts, pts);
    frame->pts = pts;
    NELUX_DEBUG("Frame PTS set to {}", pts);
}

/**
 * @brief Check if the Frame holds a valid AVFrame.
 *
 * @return True if the AVFrame is valid, False otherwise.
 */
Frame::operator bool() const
{
    bool isValid = frame != nullptr;
    NELUX_TRACE("Frame::operator bool() called: isValid = {}", isValid);
    return isValid;
}

/**
 * @brief Allocate buffer for the AVFrame with the specified alignment.
 *
 * @param align Alignment in bytes (e.g., 32 for SIMD optimizations).
 * @throws FFException if buffer allocation fails.
 */
void Frame::allocateBuffer(int align)
{
    NELUX_DEBUG("Frame Class: Frame::allocateBuffer() called with alignment = {}",
                align);
    if (av_frame_get_buffer(frame, align) < 0)
    {
        NELUX_ERROR("Failed to allocate buffer for AVFrame with alignment {}", align);
        throw CxException("Failed to allocate buffer for AVFrame with alignment " +
                          std::to_string(align));
    }
    NELUX_DEBUG("Buffer allocated for AVFrame with alignment {}", align);
}

/**
 * @brief Copy frame data from another Frame.
 *
 * @param other The source Frame to copy data from.
 * @throws FFException if frame copying fails.
 */
void Frame::copyFrom(const Frame& other)
{
    NELUX_DEBUG(
        "Frame Class: Frame::copyFrom() called: Copying data from another Frame");
    if (av_frame_copy(frame, other.frame) < 0)
    {
        NELUX_ERROR("Failed to copy data from source AVFrame");
        throw CxException("Failed to copy data from source AVFrame");
    }
    NELUX_DEBUG("Frame data copied successfully from source AVFrame");

    if (av_frame_copy_props(frame, other.frame) < 0)
    {
        NELUX_ERROR("Failed to copy properties from source AVFrame");
        throw CxException("Failed to copy properties from source AVFrame");
    }
    NELUX_DEBUG("Frame properties copied successfully from source AVFrame");
}

/**
 * @brief Fill the frame with raw data.
 *
 * @param data Pointer to the raw data buffer.
 * @param size Size of the data buffer in bytes.
 * @param plane Plane index to fill.
 * @throws FFException if data copying fails or plane index is invalid.
 */
void Frame::fillData(uint8_t* data, int size, int plane)
{
    NELUX_TRACE("Frame::fillData() called with plane = {}, size = {}", plane, size);
    if (plane < 0 || plane >= AV_NUM_DATA_POINTERS)
    {
        NELUX_ERROR("Invalid plane index in fillData(): {}", plane);
        throw CxException("Invalid plane index: " + std::to_string(plane));
    }

    int planeHeight = (plane == 0) ? getHeight() : (getHeight() + 1) / 2;
    int maxSize = frame->linesize[plane] * planeHeight;
    NELUX_DEBUG("Plane {}: planeHeight = {}, maxSize = {}", plane, planeHeight,
                maxSize);

    if (size > maxSize)
    {
        NELUX_ERROR("Data size {} exceeds buffer capacity {} for plane {}", size,
                    maxSize, plane);
        throw CxException("Data size exceeds buffer capacity for plane " +
                          std::to_string(plane));
    }

    memcpy(frame->data[plane], data, size);
    NELUX_DEBUG("Frame Class: Data filled into plane {} successfully, size = {}", plane,
                size);
}

AVColorSpace Frame::getColorSpace() const
{
    // If the AVFrame hasn't set colorspace, it is typically AVCOL_SPC_UNSPECIFIED (0).
    // Return it as-is, or do a direct guess here. We'll just return it directly for
    // clarity:
    return static_cast<AVColorSpace>(frame->colorspace);
}

AVColorRange Frame::getColorRange() const
{
    // If the AVFrame hasn't set color_range, it is typically AVCOL_RANGE_UNSPECIFIED
    // (0). Return it as-is. (We'll do the "guessing" in a separate method.)
    return static_cast<AVColorRange>(frame->color_range);
}

Frame::ColorInfo Frame::getOrGuessColorInfo() const
{
    ColorInfo info;

    // 1) Grab what the frame actually says:
    info.space = static_cast<AVColorSpace>(frame->colorspace);
    info.range = static_cast<AVColorRange>(frame->color_range);

    // 2) If unspecified, guess color space based on resolution
    //    This is a typical heuristic: if width > 1280 or height > 576 => BT.709, else
    //    BT.601 Real logic depends on your environment, so adjust as needed.
    if (info.space == AVCOL_SPC_UNSPECIFIED)
    {
        if (frame->width > 1280 || frame->height > 576)
        {
            info.space = AVCOL_SPC_BT709; // HD content
        }
        else
        {
            info.space =
                AVCOL_SPC_BT470BG; // or AVCOL_SPC_SMPTE170M, typical SD content
        }
    }

    // 3) If unspecified range, guess limited range (MPEG) for typical broadcast or
    // camera captures.
    //    Some screen captures or JPEG-based inputs might be full range.
    if (info.range == AVCOL_RANGE_UNSPECIFIED)
    {
        // The default guess is often limited range (16-235)
        info.range = AVCOL_RANGE_MPEG;
    }

    return info;
}

/**
 * @brief Overload the << operator to print Frame information.
 *
 * @param os Output stream.
 * @param frame The Frame object to print.
 * @return Reference to the output stream.
 */
std::ostream& operator<<(std::ostream& os, const Frame& frame)
{
    NELUX_TRACE("operator<< called to output Frame information");
    os << "Frame(width=" << frame.getWidth() << ", height=" << frame.getHeight()
       << ", format=" << frame.getPixelFormatString() << ", pts=" << frame.getPts()
       << ")";
    return os;
}

} // namespace nelux
